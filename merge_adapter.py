"""
Merge a trained LoRA / QLoRA adapter back into the base model weights.

Usage:
  python merge_adapter.py \
      --base_model ./models/Qwen3-8B \
      --adapter_path ./output/qlora_5060ti \
      --output_path ./models/Qwen3-8B-Poetry
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    p.add_argument("--base_model",   type=str, required=True, help="Path to base Qwen3-8B model")
    p.add_argument("--adapter_path", type=str, required=True, help="Path to trained LoRA adapter")
    p.add_argument("--output_path",  type=str, required=True, help="Output path for merged model")
    p.add_argument("--torch_dtype",  type=str, default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    args = p.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.torch_dtype]

    print(f"Loading base model from {args.base_model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading adapter from {args.adapter_path} ...")
    model = PeftModel.from_pretrained(model, args.adapter_path)

    print("Merging weights ...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_path} ...")
    model.save_pretrained(args.output_path, safe_serialization=True)
    tokenizer.save_pretrained(args.output_path)

    print("Done!")


if __name__ == "__main__":
    main()
