# NekooBasho

![NekooBashoo](slidev/image/logo.jpg)

A Regulated Verse Generation Framework for LLMs

## Features

- **Token-free Pruning**: Prune Qwen3-8B to only output single Chinese character tokens
- **Constrained Decoding**: Balancing formal constraints and literary merit through our specially designed logit processor
- **Interactive Web UI**: Gradio-based app interface for regulated verse generation (see teaser below)

<p align="center">
  <img src="slidev/image/teaser.jpg" alt="Web UI Teaser" width="64%" />
</p>

## Dependencies

Before using the system, you need to setup the Python environment.

```bash
pip install -r requirements.txt
```

For **inference only**, the minimal set is:

```bash
pip install torch transformers accelerate OpenCC
```

For the **Gradio web UI**, additionally install:

```bash
pip install gradio
```

For **fine-tuning**, you also need:

```bash
pip install datasets peft bitsandbytes trl wandb
# Multi-GPU training (optional)
pip install deepspeed
```

And then, you need to download the [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) model to the local directory `NekooBasho/ckpt/Qwen3-8B/`.

```bash
cd NekooBasho
mkdir -p ckpt
cd ckpt
git lfs clone https://huggingface.co/Qwen/Qwen3-8B
```

**Note**: The model is approximately 16GB in size. Ensure you have sufficient disk space and a stable internet connection.

## Usage

### 1. Identify Single Character Tokens

After downloading the model, run the script to identify single character tokens in Qwen3-8B tokenizer:

```bash
python -m model.prune_tokenizer
```

Or specify a custom model path:

```bash
python -m model.prune_tokenizer --model_path ./ckpt/Qwen3-8B --output_path ./ckpt/single_char_tokens.json
```

This will generate `single_char_tokens.json` in `./ckpt/`.

### 2. Generate Poetry using Pretrained Models

#### Command Line

Use the CLI to generate poetry:

```bash
python cli.py --user_prompt "春天" --poem_type "五言绝句"
```

Parameters:
- `--model_path`: Local model path (default: ./ckpt/Qwen3-8B)
- `--config_path`: Token-free config file path (default: ./ckpt/single_char_tokens.json)
- `--user_prompt`: User prompt
- `--poem_type`: Poetry type (五言绝句, 七言律诗, etc.)
- `--device`: Device (cuda/cpu)

#### Gradio Web UI

Run the web interface for interactive poetry generation:

```bash
python app.py
```

This launches a Gradio app in your browser. The UI supports:

- Theme input
- Poetry type, metrical pattern, and rhyme group selection
- Output in simplified or traditional Chinese

#### Agent

```bash
python agent_cli.py \
  --model_path /workspace/lp/ckpt/Qwen3-8B \
  --user_prompt "春天" \
  --poem_type "五言绝句" \
  --verbose \
  --device cuda
```

#### Supported Poetry Types

- 五言绝句
- 五言律诗
- 七言绝句
- 七言律诗

### 3. Prepare Fine-tuning Dataset

From the project root, run in order:

**Step 1. Download raw 全唐诗 dataset from [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)**

```bash
python -m data.download_raw_dataset
# Or: python -m data.download_raw_dataset --data_dir ./data/chinese-poetry
```

**Step 2. Filter poems and save intermediate list**

```bash
python -m data.filter_poems --output ./data/training/filtered_poems.json
```

**Step 3. Extract a short theme per poem using local Qwen3-8B model**

```bash
python -m data.extract_themes
# Optional:
#   --batch_size 32
#   --limit 64          # Test on the first 64 poems
```

**Step 4. Formats with the chat template and build final train / eval JSONL**

```bash
python -m data.build_dataset
# Optional: --output_dir ./data/training --eval_ratio 0.05 --seed 42
```

**Step 5. (Optional) Upload the dataset to Hugging Face Hub**

Upload your constructed `train.jsonl` and `eval.jsonl` files to your Hugging Face dataset repo.
First, make sure you are logged in (`huggingface-cli login`) or set the `HF_TOKEN` environment variable.

```bash
python -m data.upload_hf <your-hf-username>/<repo-name>
# Options:
#   --data_dir ./data/training    # Directory with train.jsonl and eval.jsonl (default)
#   --dry_run                     # Just preview the dataset, don't push
#   --private                     # Create repo as private
```

Example:

```bash
python -m data.upload_hf myusername/nekoobasho-dataset --private
```

This will push the dataset to https://huggingface.co/datasets/myusername/nekoobasho-dataset

### (Optional) EDA: Strict Regulated Verse Rate

Use `PoemStructureChecker` to estimate how many poems in 全唐诗 strictly satisfy rhyme + ping-ze meter rules:

```bash
python -m data.eda
```

### 4. Fine-tune on Complete Tang Poetry

Use a single config file `configs/finetune.yaml`. Adjust the parameters per your hardware; see inline comments for details.

```bash
python finetune.py --config configs/finetune.yaml
```

Multi-GPU: `torchrun --nproc_per_node=4 finetune.py --config configs/finetune.yaml` (set `deepspeed` to `"./configs/ds_zero2.yaml"` in the config first).

Resume from checkpoint: `python finetune.py --config configs/finetune.yaml --resume_from_checkpoint ckpt/lora/checkpoint-1000`

### 5. Merge LoRA Adapter

After training, merge the adapter weights into the base model to produce a standalone model:

```bash
python merge_adapter.py \
    --base_model ./ckpt/Qwen3-8B \
    --adapter_path ./ckpt/lora/checkpoint-1000 \
    --output_path ./ckpt/Qwen3-8B-Poetry
```

The merged model can then be used with `python cli.py` by passing `--model_path ./ckpt/Qwen3-8B-Poetry`.

### 6. Evaluation

Poetry generation evaluation in [CharPoet](https://arxiv.org/abs/2401.03512) style.

**Theme files** (one item per line):
- `eval/eval_idioms.txt`: 100 idioms (keyword setting).
- `eval/eval_instructions.txt`: 100 natural-language instructions.

**Step 1. Generate poems**

```bash
python eval/generate_for_eval.py
```
Output: `eval/eval_poems.json`.

**Step 2. Formal restriction stats**

```bash
python eval/formal_stats.py
```
Reads `eval/eval_poems.json`, prints format accuracy and Pingshui 格律 (rhyming_ok, meter_ok, both_ok).

**Step 3. Content quality scores**

In this step, we call Moonshot API to score generated poems to avoid human bias. Set `MOONSHOT_API_KEY` (get from [Moonshot AI](https://platform.moonshot.ai)).
```bash
set MOONSHOT_API_KEY=your_key
python eval/content_scores.py
```
Output: `eval/content_scores.json` and per-dimension means (Fluency, Meaning, Coherence, Relevance, Aesthetics, 1–5). Options: `--resume`, `--limit N`, `--delay 0.5`.

## Code Structure

```
NekooBasho/
├── ckpt/                        # Checkpoints and config (gitignored)
│   ├── Qwen3-8B/                # Downloaded Qwen3-8B model
│   ├── single_char_tokens.json  # Single-char token config (from model.prune_tokenizer)
│   └── lora/                    # LoRA fine-tuning checkpoints (from finetune.py)
├── model/                       # Model and generation logic
│   ├── __init__.py
│   ├── token_free_model.py      # Token-free Qwen3 wrapper
│   ├── utils.py                 # Poetry templates, metrical patterns, etc.
│   ├── generation.py            # load_token_free_model, generate_poem (core logic)
│   └── prune_tokenizer.py       # Build single_char_tokens.json
├── data/
│   ├── __init__.py
│   ├── utils.py                 # Download / load chinese-poetry helpers
│   ├── download_raw_dataset.py  # Download 全唐诗 only
│   ├── filter_poems.py          # Filter to regulated verse
│   ├── extract_themes.py        # Theme extraction via local Qwen3-8B
│   ├── build_dataset.py         # Build train/eval JSONL from themes
│   ├── upload_hf.py             # Upload dataset to Hugging Face Hub
│   ├── eda.py                   # Dataset EDA / statistics
│   ├── chinese-poetry/          # Downloaded chinese-poetry repo
│   └── training/                # Prepared JSONL (filtered_poems, poem_themes, train / eval)
├── eval/                        # CharPoet-style evaluation
│   ├── eval_idioms.txt          # Evaluation themes: idioms
│   ├── eval_instructions.txt    # Evaluation themes: instructions
│   ├── generate_for_eval.py     # Batch generate, output eval_poems.json
│   ├── formal_stats.py          # Formal restriction stats
│   └── content_scores.py        # 5-dim scoring, output content_scores.json
├── configs/                     # Fine-tuning configs
│   ├── finetune.yaml            #   Single config (params for different hardware)
│   ├── ds_zero2.yaml            #   DeepSpeed ZeRO-2 (multi-GPU)
│   └── ds_zero3.yaml            #   DeepSpeed ZeRO-3 (multi-GPU)
├── cli.py                       # CLI entry of poetry generation
├── app.py                       # Gradio web UI entry of poetry generation
├── finetune.py                  # LoRA / QLoRA fine-tuning script
├── merge_adapter.py             # Merge LoRA adapter into base model
├── requirements.txt             # Python dependencies
└── README.md
```
