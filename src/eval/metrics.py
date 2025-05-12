from __future__ import annotations
from typing import Dict, Any

# --------------------------------------------------------------------------- #
#  Exact Match
# --------------------------------------------------------------------------- #


def exact_match(predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> bool:
    """
    Strict string/JSON equality.
    """
    # `search_query` text is compared verbatim – paper disregarded semantic sim.
    return predicted == ground_truth


# --------------------------------------------------------------------------- #
#  AST similarity (paper §3.3, Fig 5)
# --------------------------------------------------------------------------- #


def ast_similarity(predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """
    Weighted component score (collection 40 %, search/filter/agg/group 15 % each).
    TODO: implement hierarchical compare as per Appendix C.
    """
    # TODO: implement
    raise NotImplementedError
