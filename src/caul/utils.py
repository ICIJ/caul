def fuzzy_match(key: str, candidates: set[str]) -> set[str]:
    if key in candidates:
        return {key}
    fuzzy_matches = set(k for k in candidates if key in k or k in key)
    return fuzzy_matches
