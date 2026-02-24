"""
LLMPoet Gradio Web UI
Generates Chinese poetry with configurable parameters and multi-language interface.
"""
import json
import os
import re

import torch
import gradio as gr

from generate import load_token_free_model, generate_poem
from utils import masked_poem_dict, metrical_patterns, PING_RHYME_GROUP_NAMES

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CURRENT_DIR, "single_char_tokens.json")

# UI strings: en, zh-CN, zh-TW
UI_STRINGS = {
    "en": {
        "title": "LLMPoet - Chinese Poetry Generator",
        "user_prompt": "Theme / Topic",
        "user_prompt_placeholder": "e.g., spring, mountains, friendship",
        "poem_type": "Poetry Type",
        "variant": "Metrical Pattern",
        "rhyme_group": "Rhyme Group",
        "script": "Output Script",
        "generate": "Generate",
        "output_label": "Generated Poem",
        "loading": "Loading model...",
        "error": "Error",
        "empty_prompt_error": "Please enter a theme.",
        "generation_error_prefix": "Generation failed: ",
        "random": "Random",
        "simplified": "Simplified Chinese",
        "traditional": "Traditional Chinese",
    },
    "zh-CN": {
        "title": "LLMPoet - 中文诗歌生成",
        "user_prompt": "主题",
        "user_prompt_placeholder": "例如：春天、山水、友情",
        "poem_type": "诗体",
        "variant": "格律",
        "rhyme_group": "韵部",
        "script": "输出字体",
        "generate": "生成",
        "output_label": "生成的诗",
        "loading": "正在加载模型...",
        "error": "错误",
        "empty_prompt_error": "请输入主题。",
        "generation_error_prefix": "生成失败：",
        "random": "随机",
        "simplified": "简体中文",
        "traditional": "繁体中文",
    },
    "zh-TW": {
        "title": "LLMPoet - 中文詩歌生成",
        "user_prompt": "主題",
        "user_prompt_placeholder": "例如：春天、山水、友情",
        "poem_type": "詩體",
        "variant": "格律",
        "rhyme_group": "韻部",
        "script": "輸出字體",
        "generate": "生成",
        "output_label": "生成的詩",
        "loading": "正在載入模型...",
        "error": "錯誤",
        "empty_prompt_error": "請輸入主題。",
        "generation_error_prefix": "生成失敗：",
        "random": "隨機",
        "simplified": "簡體中文",
        "traditional": "繁體中文",
    },
}


def load_rhyme_groups():
    """Load 平声韵部 names available in single_char_tokens.json."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    rhyme_index = config.get("rhyme_index", {})
    # Only 平声韵部 that have tokens in vocabulary
    groups = [k for k in PING_RHYME_GROUP_NAMES if k in rhyme_index]
    return groups


def get_variant_names():
    return [v["name"] for v in metrical_patterns["五言绝句"]]


def build_variant_choices(random_label: str):
    return [(random_label, "Random")] + [(v, v) for v in get_variant_names()]


def build_rhyme_group_choices(random_label: str, rhyme_groups: list):
    return [(random_label, "Random")] + [(g, g) for g in rhyme_groups]


def get_poem_type_choices():
    return list(masked_poem_dict.keys())


def format_poem_grid(poem_text: str) -> str:
    """
    Format poem text as HTML grid: each character in a cell, larger font.
    Poem format: chars separated by ，。、；
    """
    if not poem_text or not poem_text.strip():
        return "<p></p>"

    # Split by punctuation
    # parts = ["春眠不覺曉", "，", "處處聞啼鳥", "。", ...]
    parts = re.split(r"([，。、；])", poem_text.strip())

    rows = []
    i = 0
    while i < len(parts):
        chunk = parts[i]
        # Skip empty or whitespace-only chunks (avoids extra blank row at end)
        if not chunk.strip():
            i += 1
            continue
        if chunk in "，。、；":
            # Punctuation: append to current row and close it
            if rows:
                rows[-1] += f'<td class="punct">{chunk}</td></tr>'
            i += 1
            continue
        # Character sequence: start new row
        cells = "".join(f'<td>{c}</td>' for c in chunk)
        rows.append(f"<tr>{cells}")
        i += 1

    # Close last row if it has no punctuation (edge case)
    if rows and not rows[-1].endswith("</tr>"):
        rows[-1] += "</tr>"

    table = f'<table class="poem-grid"><tbody>{"".join(rows)}</tbody></table>'
    css = """
    <style>
    .poem-grid { border-collapse: collapse; margin: 1em auto; font-size: 1.8rem; }
    .poem-grid td { width: 2.2em; height: 2.2em; text-align: center; vertical-align: middle;
                   border: 1px solid #ccc; padding: 0.2em; }
    .poem-grid td.punct { border: none; width: 1.2em; font-size: 1.2em; }
    </style>
    """
    return css + f'<div class="poem-container">{table}</div>'


# Global model cache
_model_cache = {"model": None, "tokenizer": None}


def get_model(device="cuda"):
    """Load model once and cache."""
    if _model_cache["model"] is None:
        _model_cache["model"], _model_cache["tokenizer"] = load_token_free_model(
            device=device
        )
    return _model_cache["model"], _model_cache["tokenizer"]


def run_generate(
    user_prompt,
    poem_type,
    variant,
    rhyme_group,
    script,
    lang,
    device="cuda",
):
    """Generate poem and return (html, error_msg, error_state).
    error_state: (None, None) | ("empty_prompt", None) | ("generation", str(e))
    """
    s = UI_STRINGS.get(lang, UI_STRINGS["en"])
    if not user_prompt or not user_prompt.strip():
        msg = s.get("empty_prompt_error", s["user_prompt"])
        return format_poem_grid(""), msg, ("empty_prompt", None)

    prefix = s.get("generation_error_prefix", "")

    try:
        model, tokenizer = get_model(device)
    except Exception as e:
        return format_poem_grid(""), prefix + "\n" + str(e), ("generation", str(e))

    # Map "Random" to None
    variant_val = None if variant == "Random" else variant
    rhyme_val = None if rhyme_group == "Random" else rhyme_group

    try:
        poem_text = generate_poem(
            model=model,
            tokenizer=tokenizer,
            user_prompt=user_prompt.strip(),
            poem_type=poem_type,
            device=device,
            variant=variant_val,
            rhyme_group=rhyme_val,
            script=script,
        )
        return format_poem_grid(poem_text), "", (None, None)
    except Exception as e:
        return format_poem_grid(""), prefix + "\n" + str(e), ("generation", str(e))


def main():
    # Load rhyme groups once at startup (without Random - added in choices)
    rhyme_group_names = load_rhyme_groups()
    poem_types = get_poem_type_choices()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with gr.Blocks(title="LLMPoet", css="""
        .poem-container { display: flex; justify-content: center; }
        .lang-row { justify-content: space-between; align-items: center; }
    """) as demo:
        with gr.Row(elem_classes="lang-row"):
            title_md = gr.Markdown("# LLMPoet - Chinese Poetry Generator")
            lang_dd = gr.Dropdown(
                choices=["English", "简体中文", "繁體中文"],
                value="English",
                label="🌐 i18n",
                scale=0,
                min_width=140,
            )

        s = UI_STRINGS["en"]

        user_prompt = gr.Textbox(
            label=s["user_prompt"],
            placeholder=s["user_prompt_placeholder"],
            lines=2,
        )
        with gr.Row():
            poem_type = gr.Dropdown(
                choices=poem_types,
                value=poem_types[0],
                label=s["poem_type"],
            )
            variant = gr.Dropdown(
                choices=build_variant_choices(s["random"]),
                value="Random",
                label=s["variant"],
            )
            rhyme_group = gr.Dropdown(
                choices=build_rhyme_group_choices(s["random"], rhyme_group_names),
                value="Random",
                label=s["rhyme_group"],
            )
            script_dd = gr.Dropdown(
                choices=[(s["simplified"], "simplified"), (s["traditional"], "traditional")],
                value="simplified",
                label=s["script"],
            )
        gen_btn = gr.Button(s["generate"])

        output_html = gr.HTML(label=s["output_label"])
        error_box = gr.Textbox(label=s["error"], visible=False, interactive=False)
        error_state = gr.State(value=(None, None))  # (error_type, error_detail)

        def clear_error():
            """Hide error box immediately when Generate is clicked."""
            return gr.update(value="", visible=False)

        def generate_fn(up, pt, v, rg, sc, lang):
            html, err, err_state = run_generate(up, pt, v, rg, sc, lang, device)
            err_update = gr.update(value=err, visible=bool(err))
            return html, err_update, err_state

        def lang_to_code(display):
            return {"English": "en", "简体中文": "zh-CN", "繁體中文": "zh-TW"}.get(display, "en")

        def format_error_for_lang(lang_code, err_type, err_detail):
            """Build localized error message from stored error state."""
            s = UI_STRINGS.get(lang_code, UI_STRINGS["en"])
            if err_type == "empty_prompt":
                return s.get("empty_prompt_error", s["user_prompt"])
            if err_type == "generation" and err_detail:
                return s.get("generation_error_prefix", "") + err_detail
            return ""

        def update_labels(display, err_state):
            lang = lang_to_code(display)
            s = UI_STRINGS.get(lang, UI_STRINGS["en"])
            err_type, err_detail = err_state or (None, None)
            if err_type:
                err_msg = format_error_for_lang(lang, err_type, err_detail)
                err_box_update = gr.update(value=err_msg, visible=True, label=s["error"])
            else:
                err_box_update = gr.update(value="", visible=False, label=s["error"])
            return (
                gr.update(value=f"# {s['title']}"),
                gr.update(label=s["user_prompt"], placeholder=s["user_prompt_placeholder"]),
                gr.update(label=s["poem_type"]),
                gr.update(
                    label=s["variant"],
                    choices=build_variant_choices(s["random"]),
                ),
                gr.update(
                    label=s["rhyme_group"],
                    choices=build_rhyme_group_choices(s["random"], rhyme_group_names),
                ),
                gr.update(
                    label=s["script"],
                    choices=[(s["simplified"], "simplified"), (s["traditional"], "traditional")],
                ),
                gr.update(value=s["generate"]),
                gr.update(label=s["output_label"]),
                err_box_update,
            )

        # On Generate click: first hide error box, then run generation
        gen_btn.click(
            fn=clear_error,
            outputs=[error_box],
        ).then(
            fn=lambda up, pt, v, rg, sc, lang_display: generate_fn(up, pt, v, rg, sc, lang_to_code(lang_display)),
            inputs=[user_prompt, poem_type, variant, rhyme_group, script_dd, lang_dd],
            outputs=[output_html, error_box, error_state],
        )

        lang_dd.change(
            fn=update_labels,
            inputs=[lang_dd, error_state],
            outputs=[
                title_md,
                user_prompt,
                poem_type,
                variant,
                rhyme_group,
                script_dd,
                gen_btn,
                output_html,
                error_box,
            ],
        )

    demo.launch()


if __name__ == "__main__":
    main()
