import json

from model.utils import PING_RHYME_GROUP_NAMES, get_poem_type_display, masked_poem_dict


SYSTEM_PROMPT = (
    "你是一位擅长近体诗创作的中文诗歌助手。"
    "你要先理解任务，再给出结构化规划，最后创作完整诗作。"
    "输出必须简洁、明确，严格遵守用户给定的诗体与约束。"
)


def _base_constraints(poem_type: str, script: str, variant: str | None, rhyme_group: str | None) -> list[str]:
    display_type = get_poem_type_display(poem_type, script)
    constraints = [
        f"必须写成{display_type}",
        f"必须严格符合模板字数：{masked_poem_dict[poem_type]}",
        "正文只输出诗句本身，不要标题、注释、解释或引号",
    ]
    if variant:
        constraints.append(f"格律变体固定为：{variant}")
    if rhyme_group:
        constraints.append(f"押韵韵部固定为：{rhyme_group}")
    return constraints


def build_rewrite_prompt(
    user_prompt: str,
    poem_type: str,
    script: str,
    variant: str | None,
    rhyme_group: str | None,
) -> str:
    payload = {
        "task": "请将用户需求改写为明确的写诗任务规格。",
        "input": {
            "user_prompt": user_prompt,
            "poem_type": poem_type,
            "script": script,
            "variant": variant,
            "rhyme_group": rhyme_group,
        },
        "requirements": [
            "输出 JSON 对象，字段必须包含：style, focus, constraints",
            "style 是 4-12 个字的风格描述",
            "focus 是一句 8-30 字的创作重点",
            "constraints 是字符串数组，包含必须遵守的硬约束",
            "不要输出 Markdown 代码块",
        ],
        "defaults": {
            "style": "典雅含蓄",
            "focus": "围绕主题写出完整意境与情感推进",
            "constraints": _base_constraints(poem_type, script, variant, rhyme_group),
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_plan_prompt(
    user_prompt: str,
    poem_type: str,
    script: str,
    rewrite_spec: dict,
) -> str:
    line_count = 4 if "绝句" in poem_type else 8
    payload = {
        "task": "基于任务规格，先做写作规划，不要直接写诗。",
        "input": {
            "user_prompt": user_prompt,
            "poem_type": get_poem_type_display(poem_type, script),
            "rewrite_spec": rewrite_spec,
        },
        "requirements": [
            "输出 JSON 对象，字段必须包含：outline, imagery, emotional_arc, closing, cautions",
            f"outline 必须有 {line_count} 条，每条对应一句的语义功能",
            "imagery 列出 2-6 个核心意象",
            "closing 是结尾处理方式",
            "cautions 是需要避免的表达",
            "不要输出 Markdown 代码块",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_draft_prompt(
    user_prompt: str,
    poem_type: str,
    script: str,
    rewrite_spec: dict,
    plan_spec: dict,
    repair_hint: str = "",
) -> str:
    display_type = get_poem_type_display(poem_type, script)
    prompt = {
        "task": "请根据规格与规划创作一首完整诗作。",
        "input": {
            "user_prompt": user_prompt,
            "poem_type": display_type,
            "rewrite_spec": rewrite_spec,
            "plan_spec": plan_spec,
            "template": masked_poem_dict[poem_type],
            "repair_hint": repair_hint,
        },
        "requirements": [
            "只输出最终诗句正文",
            "使用中文全角标点",
            "不要输出标题、说明、编号、空行、引号或额外解释",
            "尽量紧扣主题，避免空泛陈词",
        ],
    }
    if repair_hint:
        prompt["requirements"].append("优先修复 repair_hint 中指出的问题")
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def build_repair_hint(failure_reasons: list[str], variant: str | None, rhyme_group: str | None) -> str:
    parts = []
    if failure_reasons:
        parts.append("失败原因：" + "；".join(failure_reasons))
    if variant:
        parts.append(f"继续保持格律变体：{variant}")
    if rhyme_group:
        parts.append(f"继续保持韵部：{rhyme_group}")
    parts.append("重新创作整首诗，优先修复形式问题，同时保留主题和主要意境。")
    return " ".join(parts)


def available_rhyme_groups() -> tuple[str, ...]:
    return PING_RHYME_GROUP_NAMES
