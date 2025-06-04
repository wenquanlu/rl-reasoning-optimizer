# From lighteval.metrics.normalizations.math_normalizer

def remove_boxed(text: str | None) -> str:
    """
    Extract the text within a \\boxed{...} environment.
    Example:
    >>> _remove_boxed(\\boxed{\\frac{2}{3}})
    \\frac{2}{3}
    """
    if text is None:
        return ""
    try:
        if "\\boxed " in text:
            left = "\\boxed "
            assert text[: len(left)] == left
            return text[len(left) :]

        left = "\\boxed{"

        assert text[: len(left)] == left
        assert text[-1] == "}"

        return text[len(left) : -1]
    except Exception:
        return ""

def last_boxed_only_string(text: str) -> str | None:
    """Extract the last \\boxed{...} or \\fbox{...} element from a string."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
        if text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = text[idx : right_brace_idx + 1]

    return retval
