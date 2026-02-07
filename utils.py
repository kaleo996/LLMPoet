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
#
# The pattern strings encode the base "一三五不论，二四六分明" tonal skeleton:
#   Five-char lines: positions 0, 2 are flexible; 1, 3, 4 are strict
#   Seven-char lines: positions 0, 2, 4 are flexible; 1, 3, 5, 6 are strict
#
# Additional prosody constraints are applied as separate layers inside get_position_constraints():
#
#   三平尾 prevention - _detect_san_ping_wei():
#       in lines ending PP, the 3rd-to-last flexible position is forced to Z
#   三仄尾 prevention - _detect_san_ze_wei():
#       in lines ending ZZ, the 3rd-to-last flexible position is forced to P
#   孤平 prevention  - _detect_gu_ping_links() + refine_constraint():
#       handled dynamically at generation time via "gu_ping_watch_positions" metadata
#
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
            "lines": ["*Z*PZ", "*P*ZP", "*P*ZZ", "*Z*PP", "*Z*PZ", "*P*ZP", "*P*ZZ", "*Z*PP"],
            "rhyme_lines": [1, 3, 5, 7],
        },
        {
            "name": "仄起首句入韵",
            "lines": ["*Z*PP", "*P*ZP", "*P*ZZ", "*Z*PP", "*Z*PZ", "*P*ZP", "*P*ZZ", "*Z*PP"],
            "rhyme_lines": [0, 1, 3, 5, 7],
        },
        {
            "name": "平起首句不入韵",
            "lines": ["*P*ZZ", "*Z*PP", "*Z*PZ", "*P*ZP", "*P*ZZ", "*Z*PP", "*Z*PZ", "*P*ZP"],
            "rhyme_lines": [1, 3, 5, 7],
        },
        {
            "name": "平起首句入韵",
            "lines": ["*P*ZP", "*Z*PP", "*Z*PZ", "*P*ZP", "*P*ZZ", "*Z*PP", "*Z*PZ", "*P*ZP"],
            "rhyme_lines": [0, 1, 3, 5, 7],
        },
    ],
    "七言律诗": [
        {
            "name": "仄起首句不入韵",
            "lines": ["*Z*P*ZZ", "*P*Z*PP", "*P*Z*PZ", "*Z*P*ZP", "*Z*P*ZZ", "*P*Z*PP", "*P*Z*PZ", "*Z*P*ZP"],
            "rhyme_lines": [1, 3, 5, 7],
        },
        {
            "name": "仄起首句入韵",
            "lines": ["*Z*P*ZP", "*P*Z*PP", "*P*Z*PZ", "*Z*P*ZP", "*Z*P*ZZ", "*P*Z*PP", "*P*Z*PZ", "*Z*P*ZP"],
            "rhyme_lines": [0, 1, 3, 5, 7],
        },
        {
            "name": "平起首句不入韵",
            "lines": ["*P*Z*PZ", "*Z*P*ZP", "*Z*P*ZZ", "*P*Z*PP", "*P*Z*PZ", "*Z*P*ZP", "*Z*P*ZZ", "*P*Z*PP"],
            "rhyme_lines": [1, 3, 5, 7],
        },
        {
            "name": "平起首句入韵",
            "lines": ["*P*Z*PP", "*Z*P*ZP", "*Z*P*ZZ", "*P*Z*PP", "*P*Z*PZ", "*Z*P*ZP", "*Z*P*ZZ", "*P*Z*PP"],
            "rhyme_lines": [0, 1, 3, 5, 7],
        },
    ],
}


def _detect_gu_ping_links(line_pattern):
    """Detect positions vulnerable to 孤平 in a single line pattern.

    孤平 occurs in a rhyming 平-tone line when, excluding the rhyme character,
    only one 平 character remains.  This function identifies which flexible
    (`*`) positions must be conditionally constrained to prevent 孤平.

    The approach: among the flexible positions (excluding rhyme), the last
    flexible position "watches" all earlier flexible positions.  If every
    earlier position was generated as 仄, the last one is forced to 平,
    guaranteeing at least two 平 characters (one strict + one forced).

    Returns:
        dict mapping `char_in_line_idx` → list of watch positions (also
        `char_in_line_idx`).  If all characters at the watch positions
        were generated as 仄, the key position must be 平.
        Returns empty dict when no 孤平 risk exists.
    """
    # 孤平 only applies to lines whose last character is P (平-tone rhyme)
    if line_pattern[-1] != "P":
        return {}

    n = len(line_pattern)

    # Count guaranteed 平 (P) and collect flexible (*) positions, excluding rhyme
    flexible_positions = []
    guaranteed_ping = 0
    for i in range(n - 1):  # skip the rhyme position
        if line_pattern[i] == "*":
            flexible_positions.append(i)
        elif line_pattern[i] == "P":
            guaranteed_ping += 1

    # ≥ 2 guaranteed 平 already → no 孤平 risk even if all * become 仄
    if guaranteed_ping >= 2:
        return {}

    if guaranteed_ping == 1:
        if len(flexible_positions) >= 2:
            # Last flexible watches all earlier ones.
            last = flexible_positions[-1]
            earlier = flexible_positions[:-1]
            return {last: earlier}
        if len(flexible_positions) == 1:
            # Only one flexible position; if it becomes 仄, only 1 平 remains → 孤平.
            # Force it to 平 (empty watch list = vacuously true).
            return {flexible_positions[0]: []}

    # guaranteed_ping == 0 is not expected in standard regulated verse
    return {}


def _detect_san_ping_wei_position(line_pattern):
    """Detect the position needing 三平尾 prevention in a single line.

    三平尾: the last three characters are all 平, forbidden in regulated verse.

    Returns:
        The idx of the position needing 三平尾 prevention. None when no constraint needed.
    """
    n = len(line_pattern)
    if line_pattern[-3] == "*" and line_pattern[-2] == "P" and line_pattern[-1] == "P":
        return n - 3
    else:
        return None


def _detect_san_ze_wei_position(line_pattern):
    """Detect the position needing 三仄尾 prevention in a single line.

    三仄尾: the last three characters are all 仄, generally avoided in regulated verse.

    Returns:
        The idx of the position needing 三仄尾 prevention. None when no constraint needed.
    """
    n = len(line_pattern)
    if line_pattern[-3] == "*" and line_pattern[-2] == "Z" and line_pattern[-1] == "Z":
        return n - 3
    else:
        return None


def refine_constraint(constraint, generated_tones):
    """Refine a position's constraint based on previously generated tones.

    This implements the dynamic 孤平 prevention rule. When a constraint
    carries a `gu_ping_watch_positions` list and all watched positions
    were generated as 仄, the tone for the current position is tightened
    from flexible (`*`) to `ping`.

    Args:
        constraint: base per-position constraint dict (from func `get_position_constraints`).
        generated_tones: dict mapping `global_char_idx` → `"ping"` | `"ze"` for characters already generated.

    Returns:
        The possibly refined constraint dict.
    """
    if "gu_ping_watch_positions" in constraint:
        # all() on an empty iterable is True, which correctly handles the edge case
        # of a single flexible position that must always be 平.
        if all(generated_tones[pos] == "ze" for pos in constraint["gu_ping_watch_positions"]):
            constraint["tone"] = "ping"

    return constraint

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

    # Determine if the first line is a non-rhyming line (首句不入韵)
    first_line_no_rhyme = (0 not in variant["rhyme_lines"])

    constraints = []
    global_offset = 0  # global index of the starting character of the current line
    for line_idx, line_pattern in enumerate(variant["lines"]):
        # Detect further constraints for this line
        san_ping_wei  = _detect_san_ping_wei_position(line_pattern)   # 三平尾
        san_ze_wei    = _detect_san_ze_wei_position(line_pattern)     # 三仄尾
        gu_ping_links = _detect_gu_ping_links(line_pattern)           # 孤平

        for char_idx, tone_char in enumerate(line_pattern):
            is_last_char_of_line = (char_idx == len(line_pattern) - 1)
            is_rhyme = (line_idx in variant["rhyme_lines"] and is_last_char_of_line)
            constraint = {"tone": _TONE_MAP[tone_char], "is_rhyme": is_rhyme}

            # 首句不入韵, explicitly forbid first line's last character from matching the rhyme group
            if first_line_no_rhyme and line_idx == 0 and is_last_char_of_line:
                constraint["is_anti_rhyme"] = True

            # 三平尾 / 三仄尾 prevention
            if san_ping_wei == char_idx:
                constraint["tone"] = "ze"

            if san_ze_wei == char_idx:
                constraint["tone"] = "ping"

            # 孤平 prevention metadata (dynamic, needs to be applied at generation time)
            # Note: 孤平 only applies to lines ending in P; 三仄尾 applies to lines
            # ending in ZZ - they are mutually exclusive, so only san_ping_wei can
            # conflict with gu_ping_links here.
            if char_idx in gu_ping_links:
                if san_ping_wei == char_idx:
                    # Conflict: 三平尾 prevention locks this (last) flex position to ze.
                    # Shift the gu_ping watch to the second-to-last flex position,
                    # which dynamically watches all earlier flex positions.
                    earlier = gu_ping_links[char_idx]  # flex positions before this one
                    if len(earlier) >= 1:
                        new_watcher_local = earlier[-1]
                        new_watch_local = earlier[:-1]
                        new_watcher_global = global_offset + new_watcher_local
                        new_watch_global = [global_offset + w for w in new_watch_local]
                        constraints[new_watcher_global]["gu_ping_watch_positions"] = new_watch_global
                else:
                    watch_local = gu_ping_links[char_idx]
                    watch_global = [global_offset + w for w in watch_local]
                    constraint["gu_ping_watch_positions"] = watch_global
            constraints.append(constraint)

        global_offset += len(line_pattern)

    return constraints, variant["name"]
