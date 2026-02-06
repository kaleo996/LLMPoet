import random

# Poetry template dictionary, using mask markers to indicate positions to fill
masked_poem_dict = {
    "五言绝句":
    "<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。",
    "五言律诗":
    "<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。",
    "七言绝句":
    "<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。",
    "七言律诗":
    "<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。",
    "菩萨蛮":
    "<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。",
    "沁园春":
    "<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|>。",
    "清平乐":
    "<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。",
    "如梦令":
    "<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|>，<|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。",
    "蝶恋花":
    "<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。",
    "水调歌头":
    "<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。",
    "卜算子":
    "<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。",
    "减字木兰花":
    "<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。",
    "满江红":
    "<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|>、<|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|>、<|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|>、<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>。<|extra_1|><|extra_1|><|extra_1|>、<|extra_1|><|extra_1|><|extra_1|><|extra_1|><|extra_1|>，<|extra_1|><|extra_1|><|extra_1|>。"
}

poetry_prompt_template = '''<|im_start|>user
请创作一首{poem_type}，主题：{user_prompt}。
请按照以下模板填写，每个<|extra_1|>位置填入一个汉字：
{masked_poem}<|im_end|>
<|im_start|>assistant
现在为您创作一首主题为“{user_prompt}”的{poem_type}：
'''

# ---------------------------------------------------------------------------
# Metrical patterns for regulated verse (近体诗)
# ---------------------------------------------------------------------------
# Each tone pattern string uses:
#   P = ping (level tone, 平声), Z = ze (oblique tone, 仄声), * = flexible (any)
# Flexible positions follow the "一三五不论，二四六分明" rule:
#   Five-char lines: positions 0, 2 are flexible; 1, 3, 4 are strict
#   Seven-char lines: positions 0, 2, 4 are flexible; 1, 3, 5, 6 are strict
# rhyme_lines lists the 0-indexed line numbers whose LAST character must rhyme.
# All regulated verse uses ping-tone (平声) rhyme.
# ---------------------------------------------------------------------------
metrical_patterns = {
    "五言绝句": [
        {
            "name": "仄起首句不入韵",
            "lines": ["*Z*PZ", "*P*ZP", "*P*ZZ", "*Z*PP"],
            "rhyme_lines": [1, 3],
        },
        {
            "name": "仄起首句入韵",
            "lines": ["*Z*PP", "*P*ZP", "*P*ZZ", "*Z*PP"],
            "rhyme_lines": [0, 1, 3],
        },
        {
            "name": "平起首句不入韵",
            "lines": ["*P*ZZ", "*Z*PP", "*Z*PZ", "*P*ZP"],
            "rhyme_lines": [1, 3],
        },
        {
            "name": "平起首句入韵",
            "lines": ["*P*ZP", "*Z*PP", "*Z*PZ", "*P*ZP"],
            "rhyme_lines": [0, 1, 3],
        },
    ],
    "七言绝句": [
        {
            "name": "仄起首句不入韵",
            "lines": ["*Z*P*ZZ", "*P*Z*PP", "*P*Z*PZ", "*Z*P*ZP"],
            "rhyme_lines": [1, 3],
        },
        {
            "name": "仄起首句入韵",
            "lines": ["*Z*P*ZP", "*P*Z*PP", "*P*Z*PZ", "*Z*P*ZP"],
            "rhyme_lines": [0, 1, 3],
        },
        {
            "name": "平起首句不入韵",
            "lines": ["*P*Z*PZ", "*Z*P*ZP", "*Z*P*ZZ", "*P*Z*PP"],
            "rhyme_lines": [1, 3],
        },
        {
            "name": "平起首句入韵",
            "lines": ["*P*Z*PP", "*Z*P*ZP", "*Z*P*ZZ", "*P*Z*PP"],
            "rhyme_lines": [0, 1, 3],
        },
    ],
    "五言律诗": [
        {
            "name": "仄起首句不入韵",
            "lines": ["*Z*PZ", "*P*ZP", "*P*ZZ", "*Z*PP",
                       "*Z*PZ", "*P*ZP", "*P*ZZ", "*Z*PP"],
            "rhyme_lines": [1, 3, 5, 7],
        },
        {
            "name": "仄起首句入韵",
            "lines": ["*Z*PP", "*P*ZP", "*P*ZZ", "*Z*PP",
                       "*Z*PZ", "*P*ZP", "*P*ZZ", "*Z*PP"],
            "rhyme_lines": [0, 1, 3, 5, 7],
        },
        {
            "name": "平起首句不入韵",
            "lines": ["*P*ZZ", "*Z*PP", "*Z*PZ", "*P*ZP",
                       "*P*ZZ", "*Z*PP", "*Z*PZ", "*P*ZP"],
            "rhyme_lines": [1, 3, 5, 7],
        },
        {
            "name": "平起首句入韵",
            "lines": ["*P*ZP", "*Z*PP", "*Z*PZ", "*P*ZP",
                       "*P*ZZ", "*Z*PP", "*Z*PZ", "*P*ZP"],
            "rhyme_lines": [0, 1, 3, 5, 7],
        },
    ],
    "七言律诗": [
        {
            "name": "仄起首句不入韵",
            "lines": ["*Z*P*ZZ", "*P*Z*PP", "*P*Z*PZ", "*Z*P*ZP",
                       "*Z*P*ZZ", "*P*Z*PP", "*P*Z*PZ", "*Z*P*ZP"],
            "rhyme_lines": [1, 3, 5, 7],
        },
        {
            "name": "仄起首句入韵",
            "lines": ["*Z*P*ZP", "*P*Z*PP", "*P*Z*PZ", "*Z*P*ZP",
                       "*Z*P*ZZ", "*P*Z*PP", "*P*Z*PZ", "*Z*P*ZP"],
            "rhyme_lines": [0, 1, 3, 5, 7],
        },
        {
            "name": "平起首句不入韵",
            "lines": ["*P*Z*PZ", "*Z*P*ZP", "*Z*P*ZZ", "*P*Z*PP",
                       "*P*Z*PZ", "*Z*P*ZP", "*Z*P*ZZ", "*P*Z*PP"],
            "rhyme_lines": [1, 3, 5, 7],
        },
        {
            "name": "平起首句入韵",
            "lines": ["*P*Z*PP", "*Z*P*ZP", "*Z*P*ZZ", "*P*Z*PP",
                       "*P*Z*PZ", "*Z*P*ZP", "*Z*P*ZZ", "*P*Z*PP"],
            "rhyme_lines": [0, 1, 3, 5, 7],
        },
    ],
}

_TONE_MAP = {"P": "ping", "Z": "ze", "*": "*"}


def get_position_constraints(poem_type, variant_name=None):
    """
    Generate per-position tone/rhyme constraints for a given poem type.

    Args:
        poem_type: e.g. "五言绝句"
        variant_name: select a variant by its human-readable name, e.g. "仄起首句不入韵".

    Returns:
        constraints: list of dicts, one per character position:
            {"tone": "ping"|"ze"|"*", "is_rhyme": bool}
            Returns None if the poem type has no metrical pattern.
        variant_name: human-readable name of the chosen variant, or None.
    """
    if poem_type not in metrical_patterns:
        return None, None

    variants = metrical_patterns[poem_type]

    # Resolve variant by name if provided
    if variant_name is not None:
        matched = False
        for i, variant in enumerate(variants):
            if variant["name"] == variant_name:
                variant_idx = i
                matched = True
                break
        if not matched:
            available = [variant["name"] for variant in variants]
            raise ValueError(f"Unknown variant '{variant_name}' for {poem_type}. Available variants: {available}")
    else: # random selection if no variant name is provided
        variant_idx = random.randint(0, len(variants) - 1)

    variant = variants[variant_idx]

    constraints = []
    for line_idx, line_pattern in enumerate(variant["lines"]):
        for char_idx, tone_char in enumerate(line_pattern):
            is_rhyme = (line_idx in variant["rhyme_lines"] and char_idx == len(line_pattern) - 1)
            constraints.append({"tone": _TONE_MAP[tone_char], "is_rhyme": is_rhyme})

    return constraints, variant["name"]
