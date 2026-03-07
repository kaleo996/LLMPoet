from .loop import run_agent_loop
from .schemas import AgentResult


def generate_poem_with_agent(
    user_prompt: str,
    poem_type: str,
    script: str = "simplified",
    variant: str | None = None,
    rhyme_group: str | None = None,
    max_rounds: int = 3,
    num_candidates: int = 3,
    device: str = "cuda",
    model_path: str | None = "./ckpt/Qwen3-8B",
) -> AgentResult:
    return run_agent_loop(
        user_prompt=user_prompt,
        poem_type=poem_type,
        script=script,
        variant=variant,
        rhyme_group=rhyme_group,
        max_rounds=max_rounds,
        num_candidates=num_candidates,
        device=device,
        model_path=model_path,
    )
