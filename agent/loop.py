import json
from typing import Any

from model.utils import metrical_patterns

from .evaluator import evaluate_poem, normalize_poem_text
from .model_runtime import generate_text, load_local_model
from .prompts import (
    SYSTEM_PROMPT,
    build_draft_prompt,
    build_plan_prompt,
    build_repair_hint,
    build_rewrite_prompt,
)
from .schemas import AgentResult, DraftAttempt, PlanSpec, RewriteSpec


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if "```" in text:
        parts = [part for part in text.split("```") if part.strip()]
        text = parts[-1].strip()
        if text.startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}


def _default_rewrite(
    user_prompt: str,
    poem_type: str,
    script: str,
    variant: str | None,
    rhyme_group: str | None,
) -> RewriteSpec:
    constraints = [
        f"必须写成{poem_type}",
        "必须严格控制句数、字数和中文标点",
        "正文只输出诗句本身",
    ]
    if variant:
        constraints.append(f"格律变体：{variant}")
    if rhyme_group:
        constraints.append(f"韵部：{rhyme_group}")
    return RewriteSpec(
        user_prompt=user_prompt,
        poem_type=poem_type,
        script=script,
        style="典雅含蓄",
        focus=f"围绕“{user_prompt}”形成完整意境并避免空泛表达",
        constraints=constraints,
        variant=variant,
        rhyme_group=rhyme_group,
    )


def _default_plan(poem_type: str, user_prompt: str) -> PlanSpec:
    line_count = 4 if "绝句" in poem_type else 8
    outline = []
    if line_count == 4:
        outline = [
            "首句起景点题",
            "次句承接景象并展开",
            "第三句转入情思或动作",
            "末句收束并点明余韵",
        ]
    else:
        outline = [
            "首联起景点题",
            "首联承题铺陈",
            "颔联写景展开",
            "颔联深化意象",
            "颈联转入情思",
            "颈联补足层次",
            "尾联收束情感",
            "尾联留下余韵",
        ]
    return PlanSpec(
        outline=outline[:line_count],
        imagery=[user_prompt, "清景", "情思"],
        emotional_arc="由景入情，含蓄收束",
        closing="结尾留有余味，不直白说理",
        cautions=["避免重复意象", "避免现代口语", "避免空泛套话"],
    )


def _parse_rewrite_spec(
    text: str,
    user_prompt: str,
    poem_type: str,
    script: str,
    variant: str | None,
    rhyme_group: str | None,
) -> RewriteSpec:
    data = _extract_json_object(text)
    fallback = _default_rewrite(user_prompt, poem_type, script, variant, rhyme_group)
    if not data:
        return fallback
    return RewriteSpec(
        user_prompt=user_prompt,
        poem_type=poem_type,
        script=script,
        style=str(data.get("style") or fallback.style),
        focus=str(data.get("focus") or fallback.focus),
        constraints=[str(item) for item in data.get("constraints", fallback.constraints)],
        variant=variant,
        rhyme_group=rhyme_group,
    )


def _parse_plan_spec(text: str, poem_type: str, user_prompt: str) -> PlanSpec:
    data = _extract_json_object(text)
    fallback = _default_plan(poem_type, user_prompt)
    if not data:
        return fallback
    outline = [str(item) for item in data.get("outline", fallback.outline)]
    expected_line_count = 4 if "绝句" in poem_type else 8
    if len(outline) < expected_line_count:
        outline = (outline + fallback.outline)[:expected_line_count]
    else:
        outline = outline[:expected_line_count]
    return PlanSpec(
        outline=outline,
        imagery=[str(item) for item in data.get("imagery", fallback.imagery)],
        emotional_arc=str(data.get("emotional_arc") or fallback.emotional_arc),
        closing=str(data.get("closing") or fallback.closing),
        cautions=[str(item) for item in data.get("cautions", fallback.cautions)],
    )


def _clean_poem_response(text: str) -> str:
    text = text.strip()
    if "```" in text:
        text = text.replace("```", "").strip()
    for prefix in ("诗：", "詩：", "答案：", "输出：", "輸出："):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return normalize_poem_text(text)


def _variant_supported(poem_type: str, variant: str | None) -> bool:
    if variant is None:
        return True
    for item in metrical_patterns.get(poem_type, []):
        if item["name"] == variant:
            return True
    return False


def run_agent_loop(
    *,
    user_prompt: str,
    poem_type: str,
    script: str = "simplified",
    variant: str | None = None,
    rhyme_group: str | None = None,
    max_rounds: int = 3,
    num_candidates: int = 3,
    device: str = "cuda",
    model_path: str | None = None,
) -> AgentResult:
    if poem_type not in metrical_patterns:
        raise ValueError(f"Unsupported poem_type: {poem_type}")
    if not _variant_supported(poem_type, variant):
        raise ValueError(f"Unknown variant '{variant}' for {poem_type}")

    model, tokenizer = load_local_model(model_path=model_path, device=device)

    rewrite_text = generate_text(
        model,
        tokenizer,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_rewrite_prompt(user_prompt, poem_type, script, variant, rhyme_group),
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.9,
    )
    rewrite_spec = _parse_rewrite_spec(
        rewrite_text, user_prompt, poem_type, script, variant, rhyme_group
    )

    plan_text = generate_text(
        model,
        tokenizer,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_plan_prompt(user_prompt, poem_type, script, rewrite_spec.to_dict()),
        max_new_tokens=384,
        temperature=0.3,
        top_p=0.9,
    )
    plan_spec = _parse_plan_spec(plan_text, poem_type, user_prompt)

    attempts: list[DraftAttempt] = []
    best_attempt: DraftAttempt | None = None
    repair_hint = ""

    for round_index in range(max_rounds):
        round_best_failed: DraftAttempt | None = None
        passed_attempts: list[DraftAttempt] = []

        for candidate_index in range(num_candidates):
            draft_prompt = build_draft_prompt(
                user_prompt,
                poem_type,
                script,
                rewrite_spec.to_dict(),
                plan_spec.to_dict(),
                repair_hint=repair_hint,
            )
            draft_text = generate_text(
                model,
                tokenizer,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=draft_prompt,
                max_new_tokens=256,
                temperature=min(0.95, 0.65 + candidate_index * 0.1),
                top_p=0.92,
            )
            poem_text = _clean_poem_response(draft_text)
            report = evaluate_poem(poem_text, poem_type, user_prompt=user_prompt)
            attempt = DraftAttempt(
                round_index=round_index,
                candidate_index=candidate_index,
                poem_text=poem_text,
                evaluation_report=report,
                repair_hint_used=repair_hint,
            )
            attempts.append(attempt)

            if report.passed:
                passed_attempts.append(attempt)
            elif (
                round_best_failed is None
                or report.quality_score > round_best_failed.evaluation_report.quality_score
            ):
                round_best_failed = attempt

        if passed_attempts:
            best_attempt = max(
                passed_attempts,
                key=lambda item: item.evaluation_report.quality_score,
            )
            break

        if round_best_failed is not None and (
            best_attempt is None
            or round_best_failed.evaluation_report.quality_score
            > best_attempt.evaluation_report.quality_score
        ):
            best_attempt = round_best_failed

        failure_reasons = (
            round_best_failed.evaluation_report.failure_reasons
            if round_best_failed is not None
            else ["unknown_generation_failure"]
        )
        repair_hint = build_repair_hint(failure_reasons, variant, rhyme_group)
        if any(reason.startswith("theme_") for reason in failure_reasons):
            plan_text = generate_text(
                model,
                tokenizer,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_plan_prompt(user_prompt, poem_type, script, rewrite_spec.to_dict()),
                max_new_tokens=384,
                temperature=0.35,
                top_p=0.9,
            )
            plan_spec = _parse_plan_spec(plan_text, poem_type, user_prompt)

    if best_attempt is None:
        empty_report = evaluate_poem("", poem_type, user_prompt=user_prompt)
        return AgentResult(
            success=False,
            poem_text="",
            rewrite_spec=rewrite_spec,
            plan_spec=plan_spec,
            evaluation_report=empty_report,
            attempts=attempts,
        )

    return AgentResult(
        success=best_attempt.evaluation_report.passed,
        poem_text=best_attempt.poem_text,
        rewrite_spec=rewrite_spec,
        plan_spec=plan_spec,
        evaluation_report=best_attempt.evaluation_report,
        attempts=attempts,
    )
