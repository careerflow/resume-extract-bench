import jellyfish


def normalize_text(text: str | None) -> str:
    """Normalize text for comparison - lowercase, strip whitespace, remove corporate suffixes."""
    text = (text or "").lower().strip()

    for suffix in [
        " llc", " inc", " inc.", " corp", " corp.", " ltd", " ltd.",
        " co.", " gmbh", " plc",
    ]:
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()

    return text


def field_similarity(a: str | None, b: str | None) -> float:
    """Jaro-Winkler similarity after normalization. Returns 0.0-1.0."""
    a = normalize_text(a)
    b = normalize_text(b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return jellyfish.jaro_winkler_similarity(a, b)


def token_f1(gt_text: str, pred_text: str) -> float:
    """Bag-of-words F1 between two text blocks. Returns 0.0-1.0."""
    gt_tokens = set(gt_text.lower().split())
    pred_tokens = set(pred_text.lower().split())

    if not gt_tokens and not pred_tokens:
        return 1.0

    if not gt_tokens or not pred_tokens:
        return 0.0

    overlap = gt_tokens & pred_tokens
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(gt_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)
