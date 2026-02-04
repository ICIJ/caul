from typing import Any


def dict_key_fuzzy_match(dict_obj: dict, search_key: str) -> Any | None:
    """Match a dict key name fuzzily (returning first match)

    :param dict_obj: dictionary to search in
    :param search_key: key name
    :return: key value (if exists)
    """

    if search_key in dict_obj:
        return dict_obj[search_key]

    fuzzy_matches = [
        dict_value
        for dict_key, dict_value in dict_obj.items()
        if dict_key in search_key or search_key in dict_key
    ]

    if len(fuzzy_matches) > 0:
        return fuzzy_matches[0]

    return None
