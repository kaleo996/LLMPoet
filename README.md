# LLMPoet

LLM poetry generation system based on Qwen3-8B

## Features

1. **Token-free Pruning**: Prune Qwen3-8B to only output single Chinese character tokens
2. **Poetry Template Prompts**: Use mask templates to indicate character count requirements for each line
3. **Constrained Decoding**: Balancing formal constraints and literary merit through our specially designed logit processor

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

And then, you need to download the [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) model to the local directory `LLMPoet/models/Qwen3-8B/`.

```bash
cd LLMPoet
mkdir -p models
cd models
git lfs clone https://huggingface.co/Qwen/Qwen3-8B
```

**Note**: The model is approximately 16GB in size. Ensure you have sufficient disk space and a stable internet connection.

## Usage

### 1. Identify Single Character Tokens

After downloading the model, run the script to identify single character tokens in Qwen3-8B tokenizer:

```bash
python prune_tokenizer.py
```

Or specify a custom model path:

```bash
python prune_tokenizer.py --model_path ./models/Qwen3-8B
```

This will generate `single_char_tokens.json` and `token_free_config.json` files in the project root.

### 2. Generate Poetry

#### Command Line

Use the generation script to generate poetry:

```bash
python generate.py --user_prompt "春天" --poem_type "五言绝句"
```

Parameters:
- `--model_path`: Local model path (default: ./models/Qwen3-8B)
- `--config_path`: Token-free config file path (default: ./token_free_config.json)
- `--user_prompt`: User prompt
- `--poem_type`: Poetry type (五言绝句, 七言律诗, etc.)
- `--device`: Device (cuda/cpu)

#### Gradio Web UI

Run the web interface for interactive poetry generation:

```bash
python app.py
```

This launches a Gradio app in your browser. The UI supports:

- Theme/topic input
- Poetry type, metrical pattern, and rhyme group selection
- Output in simplified or traditional Chinese

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
python -m data.upload_hf myusername/llmpoet-dataset --private
```

This will push the dataset to https://huggingface.co/datasets/myusername/llmpoet-dataset

### (Optional) EDA: Strict Regulated Verse Rate

Use `PoemStructureChecker` to estimate how many poems in 全唐诗 strictly satisfy rhyme + ping-ze meter rules:

```bash
# Run from the project root (module mode).
python -m data.eda
```

### 4. Fine-tune on Complete Tang Poetry

Use a single config file `configs/finetune.yaml`. Adjust the parameters per your hardware; see inline comments for details.

```bash
python finetune.py --config configs/finetune.yaml
```

Multi-GPU: `torchrun --nproc_per_node=4 finetune.py --config configs/finetune.yaml` (set `deepspeed` to `"./configs/ds_zero2.yaml"` in the config first).

Resume from checkpoint: `python finetune.py --config configs/finetune.yaml --resume_from_checkpoint output/finetune/checkpoint-1000`

### 5. Merge LoRA Adapter

After training, merge the adapter weights into the base model to produce a standalone model:

```bash
python merge_adapter.py \
    --base_model ./models/Qwen3-8B \
    --adapter_path ./output/finetune \
    --output_path ./models/Qwen3-8B-Poetry
```

The merged model can then be used with `generate.py` by passing `--model_path ./models/Qwen3-8B-Poetry`.

## Code Structure

```
LLMPoet/
├── models/
│   └── Qwen3-8B/                # Downloaded Qwen3-8B model
├── data/
│   ├── __init__.py              # Make data a Python package
│   ├── utils.py                 # Download / load chinese-poetry helpers
│   ├── download_raw_dataset.py  # Download 全唐诗 only
│   ├── filter_poems.py          # Filter to regulated verse
│   ├── extract_themes.py        # Theme extraction via local Qwen3-8B
│   ├── build_dataset.py         # Build train/eval JSONL from themes
│   ├── upload_hf.py             # Upload dataset to Hugging Face Hub
│   ├── eda.py                   # Dataset EDA / statistics
│   ├── chinese-poetry/          # Downloaded chinese-poetry repo
│   └── training/                # Prepared JSONL (filtered_poems, poem_themes, train / eval)
├── configs/                     # Fine-tuning configs
│   ├── finetune.yaml            #   Single config (params for different hardware)
│   ├── ds_zero2.yaml            #   DeepSpeed ZeRO-2 (multi-GPU)
│   └── ds_zero3.yaml            #   DeepSpeed ZeRO-3 (multi-GPU)
├── output/                      # Training outputs (checkpoints, adapters)
├── token_free_model.py          # Token-free model wrapper class
├── utils.py                     # Utility functions (templates, formatting, etc.)
├── app.py                       # Gradio web UI
├── generate.py                  # Generation script
├── batch_generate.py            # Batch generation script
├── finetune.py                  # LoRA / QLoRA fine-tuning script
├── merge_adapter.py             # Merge LoRA adapter into base model
├── prune_tokenizer.py           # Identify single character tokens
├── requirements.txt             # Python dependencies
└── README.md
```
