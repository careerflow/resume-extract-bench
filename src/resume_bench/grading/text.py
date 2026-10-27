import re

import jellyfish

_UNICODE_PUNCT = str.maketrans({
    '‘': "'", '’': "'",   # left/right single curly quote
    '“': '"', '”': '"',   # left/right double curly quote
    '′': "'", '″': '"',   # prime / double prime
    'ʼ': "'",                  # modifier letter apostrophe
    '–': '-', '—': '-',   # en-dash / em-dash
    '―': '-', '‒': '-',   # horizontal bar / figure dash
    '‐': '-',                  # hyphen char (distinct from ASCII hyphen-minus)
})


def _normalize_unicode_punct(s: str) -> str:
    """Replace curly quotes and en/em dashes with their ASCII equivalents."""
    return s.translate(_UNICODE_PUNCT)


_BULLET_PREFIX_RE = re.compile(
    r'^[\s•●○◦▪▸►▹→⮞⁃■□✓✔✗✘★☆※‣⦁∙⚫➢➤⬥❖◆◇▶▷►‣·–—›»​‌‍﻿'
    r'\U0001f680\U0001f4cc\U0001f4cd'
    r'❖➔⮚]+\s*'
)


def _strip_bullet_prefix(s: str) -> str:
    """Strip leading bullet-point symbols and decorators from text."""
    if not s:
        return s
    return _BULLET_PREFIX_RE.sub('', s)


def normalize_text(text: str | None) -> str:
    """Normalize text for comparison - lowercase, strip whitespace, remove corporate suffixes."""
    text = (text or "").lower().strip()
    text = _normalize_unicode_punct(text)

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


def edit_distance_ratio(a: str, b: str) -> float:
    """Normalized edit distance similarity. Returns 0.0-1.0."""
    a = _strip_bullet_prefix(_normalize_unicode_punct((a or "").lower().strip()))
    b = _strip_bullet_prefix(_normalize_unicode_punct((b or "").lower().strip()))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    dist = jellyfish.levenshtein_distance(a, b)
    return 1.0 - dist / max(len(a), len(b))


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
