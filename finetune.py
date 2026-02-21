"""
Fine-tune Qwen3-8B for classical Chinese poetry generation with LoRA / QLoRA.

Supports multiple hardware configurations:
  - Quantization
  - LoRA
  - DeepSpeed on multi-GPU

Usage:
  python finetune.py --config configs/finetune.yaml

  # Multi-GPU with DeepSpeed
  torchrun --nproc_per_node=4 finetune.py --config configs/finetune.yaml

  # Resume from checkpoint
  python finetune.py --config configs/finetune.yaml \
      --resume_from_checkpoint output/finetune/checkpoint-1000
"""

import json
import math
import os
from typing import Optional

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def load_config(config_path: str, resume_from_checkpoint=None) -> dict:
    """Load config from YAML. CLI overrides only resume_from_checkpoint."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if resume_from_checkpoint is not None:
        cfg["resume_from_checkpoint"] = resume_from_checkpoint
    return cfg


def load_deepspeed_config(path: str) -> dict:
    """Load DeepSpeed config from YAML file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_quantization_config(cfg):
    q = cfg.get("quantization", "none")
    if q == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    if q == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_model_and_tokenizer(cfg):
    model_path = cfg["model_path"]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Quantization
    quant_cfg = build_quantization_config(cfg)
    use_bf16 = cfg.get("bf16", True)

    attn_impl = cfg.get("attn_implementation", "sdpa")
    model_kwargs = dict(
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        attn_implementation=attn_impl,
    )
    deepspeed_cfg = cfg.get("deepspeed")
    if quant_cfg:
        model_kwargs["quantization_config"] = quant_cfg
        model_kwargs["device_map"] = "auto"
    elif not deepspeed_cfg:
        model_kwargs["device_map"] = "auto"

    print(f"Loading base model from {model_path} ...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    except (ValueError, ImportError) as e:
        if attn_impl == "flash_attention_2":
            print(f"flash_attention_2 failed ({e}), falling back to sdpa")
            model_kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        else:
            raise

    # Prepare quantized model for training
    if quant_cfg:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        )
    elif cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    # LoRA (optional; use_lora=false for full-parameter fine-tuning)
    use_lora = cfg.get("use_lora", True)
    if use_lora:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.get("lora_rank", 64),
            lora_alpha=cfg.get("lora_alpha", 128),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=cfg.get("lora_target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]),
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        # Full-parameter fine-tuning: train all parameters
        for param in model.parameters():
            param.requires_grad = True
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Full-parameter fine-tuning: {n_params:,} trainable parameters")

    return model, tokenizer


def prepare_dataset(cfg, tokenizer):
    raw = load_dataset("json", data_files={
        "train": cfg["train_data"],
        "eval":  cfg["eval_data"],
    })

    max_len = cfg.get("max_seq_length", 512)

    # The assistant response has two parts:
    #   1. Intro (template): "現在為您創作一首主題為「xxx」的{poem_type}：\n"
    #   2. Poem content (what we want to train on)
    # We mask both the user prompt and the assistant intro, computing loss only on the poem.
    COLON_NEWLINE = "：\n"

    ASSISTANT_HEADER = "<|im_start|>assistant\n"

    def poem_start_char_offset(text: str) -> Optional[int]:
        """Return the character index where the poem body starts, or None if malformed."""
        # Directly find the last occurrence of COLON_NEWLINE to locate poem body
        pos = text.rfind(COLON_NEWLINE)
        if pos >= 0:
            return pos + len(COLON_NEWLINE)
        # Fallback: If COLON_NEWLINE not found, try finding ASSISTANT_HEADER
        idx = text.find(ASSISTANT_HEADER)
        if idx < 0:
            return None
        return idx + len(ASSISTANT_HEADER)

    def tokenize_and_mask(examples):
        tok = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_len,
            padding=False,
            return_offsets_mapping=True,
        )
        all_labels = []
        texts = examples["text"]
        if isinstance(texts, str):
            texts = [texts]
        offset_mappings = tok["offset_mapping"]

        for i, ids in enumerate(tok["input_ids"]):
            labels = list(ids)
            offsets = offset_mappings[i]
            poem_start = poem_start_char_offset(texts[i]) if i < len(texts) else None

            if poem_start is not None:
                for j, (start, end) in enumerate(offsets):
                    if j >= len(labels):
                        break
                    # Mask tokens that end before the poem starts; (0,0) = special/padding
                    if (start == 0 and end == 0) or end <= poem_start:
                        labels[j] = -100

            all_labels.append(labels)
        del tok["offset_mapping"]
        tok["labels"] = all_labels
        return tok

    dataset = raw.map(
        tokenize_and_mask,
        batched=True,
        remove_columns=raw["train"].column_names,
        num_proc=cfg.get("preprocessing_num_workers", 4),
        desc="Tokenizing",
    )
    return dataset


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-8B for classical Chinese poetry generation")
    parser.add_argument("--config", type=str, required=True, help="Path to config file (YAML)")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Resume from checkpoint directory")
    parser.add_argument("--local_rank", type=int, default=-1, help="Set by torchrun / DeepSpeed; do not pass manually")
    args = parser.parse_args()

    cfg = load_config(args.config, args.resume_from_checkpoint)

    print("Config:\n" + json.dumps(cfg, indent=2, ensure_ascii=False))

    model, tokenizer = load_model_and_tokenizer(cfg)
    dataset = prepare_dataset(cfg, tokenizer)

    print(f"Dataset — train: {len(dataset['train'])} samples, eval: {len(dataset['eval'])} samples")

    deepspeed_cfg = cfg.get("deepspeed")
    if isinstance(deepspeed_cfg, str):
        path = os.path.abspath(deepspeed_cfg)
        deepspeed_cfg = load_deepspeed_config(path)

    training_args = TrainingArguments(
        output_dir                    = cfg["output_dir"],
        num_train_epochs              = cfg.get("num_train_epochs", 3),
        per_device_train_batch_size   = cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size    = cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps   = cfg.get("gradient_accumulation_steps", 8),
        learning_rate                 = cfg.get("learning_rate", 2e-4),
        optim                         = cfg.get("optim") or ("adamw_bnb_8bit" if cfg.get("quantization") in ("4bit", "8bit") else "adamw_torch"),
        lr_scheduler_type             = cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio                  = cfg.get("warmup_ratio", 0.03),
        weight_decay                  = cfg.get("weight_decay", 0.01),
        max_grad_norm                 = cfg.get("max_grad_norm", 1.0),
        bf16                          = cfg.get("bf16", True),
        fp16                          = cfg.get("fp16", False),
        gradient_checkpointing        = cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        logging_steps                 = cfg.get("logging_steps", 10),
        eval_strategy                 = cfg.get("eval_strategy", "steps"),
        eval_steps                    = cfg.get("eval_steps", 500),
        save_strategy                 = cfg.get("save_strategy", "steps"),
        save_steps                    = cfg.get("save_steps", 500),
        save_total_limit              = cfg.get("save_total_limit", 3),
        load_best_model_at_end        = cfg.get("load_best_model_at_end", True),
        metric_for_best_model         = "eval_loss",
        greater_is_better             = False,
        report_to                     = cfg.get("report_to", "none"),
        seed                          = cfg.get("seed", 42),
        dataloader_num_workers        = cfg.get("dataloader_num_workers", 4),
        dataloader_pin_memory         = True,
        deepspeed                     = deepspeed_cfg,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8,
            label_pad_token_id=-100,
        ),
        processing_class=tokenizer,
    )

    # Train
    ckpt = cfg.get("resume_from_checkpoint")
    print(f"Starting training{' (resuming from ' + ckpt + ')' if ckpt else ''} ...")
    result = trainer.train(resume_from_checkpoint=ckpt)

    # Save
    trainer.save_model()
    trainer.save_state()

    metrics = result.metrics
    metrics["train_samples"] = len(dataset["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Eval
    print("Running final evaluation ...")
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(dataset["eval"])
    ppl = math.exp(eval_metrics.get("eval_loss", 0))
    eval_metrics["perplexity"] = ppl
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    print(f"Done. Adapter saved to {cfg['output_dir']}  |  eval_loss={eval_metrics.get('eval_loss', -1):.4f}  perplexity={ppl:.2f}")


if __name__ == "__main__":
    main()
