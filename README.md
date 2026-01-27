# LLMPoet

LLM poetry generation system based on Qwen3-8B

## Features

1. **Token-free Pruning**: Prune Qwen3-8B to only output single Chinese character tokens
2. **Poetry Template Prompts**: Use mask templates to indicate character count requirements for each line
3. **Multiple Poetry Formats**: Support for various formats like 五言绝句, 七言律诗, and 如梦令

## Dependencies

Before using the system, you need to setup the Python environment.

```bash
pip install torch transformers accelerate
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

Use the generation script to generate poetry:

```bash
python generate.py --user_prompt "Write a poem about spring" --poem_type "五言绝句"
```

Parameters:
- `--model_path`: Local model path (default: ./models/Qwen3-8B)
- `--config_path`: Token-free config file path (default: ./token_free_config.json)
- `--user_prompt`: User prompt
- `--poem_type`: Poetry type (五言绝句, 七言律诗, 如梦令, etc.)
- `--device`: Device (cuda/cpu)

### 3. Supported Poetry Types

- 五言绝句
- 五言律诗
- 七言绝句
- 七言律诗
- 菩萨蛮
- 沁园春
- 清平乐
- 如梦令
- 蝶恋花
- 水调歌头
- 卜算子
- 减字木兰花
- 满江红

## Code Structure

```
LLMPoet/
├── models/
│   └── Qwen3-8B/        # Downloaded Qwen3-8B model (you need to download this)
├── token_free_model.py  # Token-free model wrapper class
├── utils.py             # Utility functions (templates, formatting, etc.)
├── generate.py          # Generation script
├── prune_tokenizer.py   # Identify single character tokens
└── README.md
```

## Implementation Details

### Token-free Pruning

By inheriting from `Qwen3ForCausalLM` and overriding the `forward` method, we apply a mask after computing logits, keeping only single character token logits and setting others to negative infinity.

### Poetry Templates

Use `<|extra_1|>` as mask markers to indicate that each position needs to be filled with one Chinese character.
