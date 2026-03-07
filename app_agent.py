"""
LLMPoet Agent Gradio Web UI
Simple demo app for the agent-based poetry generation pipeline.
"""
import json
import re

import gradio as gr
import torch

from agent.api import generate_poem_with_agent
from model.utils import PING_RHYME_GROUP_NAMES, masked_poem_dict, metrical_patterns


def format_poem_grid(poem_text: str) -> str:
    """Format poem text as an HTML grid."""
    if not poem_text or not poem_text.strip():
        return "<p></p>"

    parts = re.split(r"([，。、；])", poem_text.strip())
    rows = []
    i = 0
    while i < len(parts):
        chunk = parts[i]
        if not chunk.strip():
            i += 1
            continue
        if chunk in "，。、；":
            if rows:
                rows[-1] += f'<td class="punct">{chunk}</td></tr>'
            i += 1
            continue
        cells = "".join(f"<td>{char}</td>" for char in chunk)
        rows.append(f"<tr>{cells}")
        i += 1

    if rows and not rows[-1].endswith("</tr>"):
        rows[-1] += "</tr>"

    return """
    <style>
    .poem-grid { border-collapse: collapse; margin: 1em auto; font-size: 1.8rem; }
    .poem-grid td { width: 2.2em; height: 2.2em; text-align: center; vertical-align: middle;
                   border: 1px solid #ccc; padding: 0.2em; }
    .poem-grid td.punct { border: none; width: 1.2em; font-size: 1.2em; }
    </style>
    """ + f'<div class="poem-container"><table class="poem-grid"><tbody>{"".join(rows)}</tbody></table></div>'


def get_variant_choices(poem_type: str) -> list[tuple[str, str]]:
    variants = metrical_patterns.get(poem_type, [])
    return [("自动", "Random")] + [(item["name"], item["name"]) for item in variants]


def get_rhyme_choices() -> list[tuple[str, str]]:
    return [("自动", "Random")] + [(item, item) for item in PING_RHYME_GROUP_NAMES]


def format_report(result) -> str:
    report = result.evaluation_report
    lines = [
        f"success: {result.success}",
        f"format_ok: {report.format_ok}",
        f"line_length_ok: {report.line_length_ok}",
        f"rhyming_ok: {report.rhyming_ok}",
        f"meter_ok: {report.meter_ok}",
        f"quality_score: {report.quality_score}",
    ]
    if report.failure_reasons:
        lines.append("failure_reasons:")
        lines.extend(f"- {item}" for item in report.failure_reasons)
    return "\n".join(lines)


def run_generate(
    user_prompt: str,
    poem_type: str,
    variant: str,
    rhyme_group: str,
    script: str,
    max_rounds: int,
    num_candidates: int,
    device: str,
):
    if not user_prompt or not user_prompt.strip():
        return format_poem_grid(""), "请输入主题。", ""

    variant_value = None if variant == "Random" else variant
    rhyme_value = None if rhyme_group == "Random" else rhyme_group

    try:
        result = generate_poem_with_agent(
            user_prompt=user_prompt.strip(),
            poem_type=poem_type,
            script=script,
            variant=variant_value,
            rhyme_group=rhyme_value,
            max_rounds=max_rounds,
            num_candidates=num_candidates,
            device=device,
        )
    except Exception as exc:
        return format_poem_grid(""), f"生成失败：{exc}", ""

    return (
        format_poem_grid(result.poem_text),
        format_report(result),
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
    )


def main():
    poem_types = list(masked_poem_dict.keys())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with gr.Blocks(
        title="LLMPoet Agent",
        css="""
        .poem-container { display: flex; justify-content: center; }
        """,
    ) as demo:
        gr.Markdown("# LLMPoet Agent Demo")

        user_prompt = gr.Textbox(
            label="主题",
            placeholder="例如：春夜、山中访友、边塞秋思",
            lines=2,
        )

        with gr.Row():
            poem_type = gr.Dropdown(
                choices=poem_types,
                value=poem_types[0],
                label="诗体",
            )
            variant = gr.Dropdown(
                choices=get_variant_choices(poem_types[0]),
                value="Random",
                label="格律变体",
            )
            rhyme_group = gr.Dropdown(
                choices=get_rhyme_choices(),
                value="Random",
                label="韵部",
            )
            script = gr.Dropdown(
                choices=[("简体中文", "simplified"), ("繁体中文", "traditional")],
                value="simplified",
                label="输出字体",
            )

        with gr.Row():
            max_rounds = gr.Slider(
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                label="最大修复轮数",
            )
            num_candidates = gr.Slider(
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                label="每轮候选数",
            )
            device_box = gr.Dropdown(
                choices=["cuda", "cpu"],
                value=device,
                label="设备",
            )

        generate_btn = gr.Button("生成")

        output_html = gr.HTML(label="诗歌")
        report_box = gr.Textbox(label="评测结果", lines=8, interactive=False)
        with gr.Accordion("调试信息", open=False):
            debug_json = gr.Code(label="Agent Result", language="json")

        poem_type.change(
            fn=lambda pt: gr.update(choices=get_variant_choices(pt), value="Random"),
            inputs=[poem_type],
            outputs=[variant],
        )

        generate_btn.click(
            fn=run_generate,
            inputs=[
                user_prompt,
                poem_type,
                variant,
                rhyme_group,
                script,
                max_rounds,
                num_candidates,
                device_box,
            ],
            outputs=[output_html, report_box, debug_json],
        )

    demo.launch()


if __name__ == "__main__":
    main()
