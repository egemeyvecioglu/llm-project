import sys
import json
from pathlib import Path
from typing import Any, Dict, List


class DBQueryScorer:
    WEIGHTS = dict(
        collection=0.40, search_query=0.15, filters=0.15, aggregation=0.15, group_by=0.15
    )
    FILTER_KEYS = {"integer_property_filter", "text_property_filter", "boolean_property_filter"}
    AGG_KEYS = {
        "integer_property_aggregation",
        "text_property_aggregation",
        "boolean_property_aggregation",
    }

    def __init__(self, gt_path: Path):
        self.gt_path = gt_path
        self.gt_idx = self._index(self._load(self.gt_path))

    @classmethod
    def _score_record(cls, gt_q: Dict[str, Any], pr_q: Dict[str, Any]):
        if not gt_q.get("target_collection") == pr_q.get("target_collection"):
            return 0, 0

        score = cls.WEIGHTS["collection"]

        if bool(gt_q.get("search_query")) == bool(pr_q.get("search_query")):
            score += cls.WEIGHTS["search_query"]

        filters_score = 0
        filters_count = 0
        for key in cls.FILTER_KEYS:
            if gt_q.get(key) == pr_q.get(key):
                filters_score += 1
            filters_count += 1

        score += cls.WEIGHTS["filters"] * (filters_score / filters_count)

        aggs_score = 0
        aggs_count = 0
        for key in cls.AGG_KEYS:
            if gt_q.get(key) == pr_q.get(key):
                aggs_score += 1
            aggs_count += 1

        score += cls.WEIGHTS["aggregation"] * (aggs_score / aggs_count)

        if gt_q.get("groupby_property") == pr_q.get("groupby_property"):
            score += cls.WEIGHTS["group_by"]

        exact = score >= 0.95
        return score, exact

    @staticmethod
    def _load(path: Path):
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data["predictions"] if isinstance(data, dict) and "predictions" in data else data

    @staticmethod
    def _index(records):
        return {r["query"]["corresponding_natural_language_query"]: r["query"] for r in records}

    def evaluate_ast_and_exact(self, pred_file: Path):
        pred_idx = self._index(self._load(pred_file))

        exact_hits = ast_total = 0.0
        gt_items = list(self.gt_idx.items())

        for nl_query, gt_q in gt_items:
            pr_q = pred_idx.get(nl_query)
            if pr_q is None:
                continue  # absent counts as miss
            ast, exact = self._score_record(gt_q, pr_q)
            ast_total += ast
            exact_hits += exact

        n = len(gt_items)

        print(f"Exact-match accuracy: {exact_hits / n:.2%}")
        print(f"Mean AST score     : {ast_total  / n:.3f}")

    def evaluate_collection_routing(self, pred_file: Path):
        pred_records = self._load(pred_file)
        gt_items = list(self.gt_idx.items())

        hits = total = 0
        pred_idx = {r["query"]["corresponding_natural_language_query"]: r for r in pred_records}
        for nl_query, gt_q in gt_items:
            rec = pred_idx.get(nl_query)
            if rec is None:
                continue
            pr_q = rec["query"]
            if gt_q.get("target_collection") == pr_q.get("target_collection"):
                hits += 1
            total += 1
        print(f"Collection Routing Accuracy: {hits / total:.2%} ({hits}/{total})")

    def evaluate_component_accuracy(self, pred_file: Path):
        pred_records = self._load(pred_file)
        gt_items = list(self.gt_idx.items())
        pred_idx = {r["query"]["corresponding_natural_language_query"]: r for r in pred_records}

        # Components
        keys = [
            "search_query",
            "integer_property_filter",
            "text_property_filter",
            "boolean_property_filter",
            "integer_property_aggregation",
            "text_property_aggregation",
            "boolean_property_aggregation",
            "groupby_property",
        ]
        hits = {k: 0 for k in keys}
        total = {k: 0 for k in keys}

        for nl_query, gt_q in gt_items:
            rec = pred_idx.get(nl_query)
            if rec is None:
                continue
            pr_q = rec["query"]

            # search_query (boolean presence)
            total["search_query"] += 1
            if bool(gt_q.get("search_query")) == bool(pr_q.get("search_query")):
                hits["search_query"] += 1

            # Each filter and aggregation (exact match)
            for k in [
                "integer_property_filter",
                "text_property_filter",
                "boolean_property_filter",
                "integer_property_aggregation",
                "text_property_aggregation",
                "boolean_property_aggregation",
            ]:
                total[k] += 1
                if gt_q.get(k) == pr_q.get(k):
                    hits[k] += 1

            # groupby_property
            total["groupby_property"] += 1
            if gt_q.get("groupby_property") == pr_q.get("groupby_property"):
                hits["groupby_property"] += 1

        for k in keys:
            acc = hits[k] / total[k] if total[k] else 0
            print(f"{k} Accuracy: {acc:.2%} ({hits[k]}/{total[k]})")

    def evaluate_no_tool_selected_rate(self, pred_file: Path):
        pred_records = self._load(pred_file)
        total = len(pred_records)
        no_tool_count = sum(1 for r in pred_records if not r.get("tool_called"))
        print(f"No Tool Selected Rate: {no_tool_count / total:.2%} ({no_tool_count}/{total})")

    def evaluate_by_complexity(self, pred_file: Path):
        pred_idx = self._index(self._load(pred_file))
        buckets = {"simple": [], "moderate": [], "complex": []}

        def _piece_count(q: Dict[str, Any]) -> int:
            """Return how many argument *pieces* the GT query uses."""
            c = 0
            if q.get("search_query"):
                c += 1
            for k in self.FILTER_KEYS:
                if q.get(k):
                    c += 1
            for k in self.AGG_KEYS:
                if q.get(k):
                    c += 1
            if q.get("groupby_property"):
                c += 1
            return c

        for nl_query, gt_q in self.gt_idx.items():
            pr_q = pred_idx.get(nl_query)

            pieces = _piece_count(gt_q)
            bucket = "simple" if pieces == 1 else "moderate" if pieces == 2 else "complex"

            # use _score_record to decide exact-match (index 1 of tuple)
            exact = self._score_record(gt_q, pr_q)[1]
            buckets[bucket].append(exact)

        for name in ("simple", "moderate", "complex"):
            total = len(buckets[name])
            acc = sum(buckets[name]) / total if total else 0
            print(f"{name.capitalize()} Exact-Match: {acc:.2%} ({sum(buckets[name])}/{total})")

    def evaluate_by_schema(self, pred_file: Path):
        schema_groups = {
            "Restaurants": ["Restaurants", "Reservations", "Menus"],
            "Health Clinics": ["Clinics", "Appointments", "Doctors"],
            "Courses": ["Courses", "Instructors", "Students"],
            "Travel Planning": ["TravelAgents", "TravelDestinations", "TravelPackages"],
            "Visual Art": ["Museums", "Exhibitions", "ArtPieces"],
        }

        pred_idx = self._index(self._load(pred_file))
        # Create reverse mapping from collection to schema group
        collection_to_group = {}
        for group, collections in schema_groups.items():
            for collection in collections:
                collection_to_group[collection] = group

        # Initialize scores per schema group
        group_scores = {group: [] for group in schema_groups}

        # Calculate exact scores for each query grouped by schema
        for nl_query, gt_q in self.gt_idx.items():
            pr_q = pred_idx.get(nl_query)
            if pr_q is None:
                continue

            target_collection = gt_q.get("target_collection")
            schema_group = collection_to_group.get(target_collection)
            if schema_group:
                exact = self._score_record(gt_q, pr_q)[1]
                group_scores[schema_group].append(exact)

        # Print results for each schema group
        for group, scores in group_scores.items():
            total = len(scores)
            if total > 0:
                exact_matches = sum(scores)
                accuracy = exact_matches / total
                print(f"{group} Schema Exact-Match: {accuracy:.2%} ({exact_matches}/{total})")

    # -------------------------------------------------
    # Convenience wrapper: run every metric in one call
    # -------------------------------------------------
    def evaluate_all(self, pred_file: Path):
        print("\n=== Exact-Match & AST ===")
        self.evaluate_ast_and_exact(pred_file)

        print("\n=== No-Tool-Selected Rate ===")
        self.evaluate_no_tool_selected_rate(pred_file)

        print("\n=== Component-wise Accuracy ===")
        self.evaluate_component_accuracy(pred_file)

        print("\n=== Collection Routing Accuracy ===")
        self.evaluate_collection_routing(pred_file)

        print("\n=== Complexity Buckets ===")
        self.evaluate_by_complexity(pred_file)

        print("\n=== Schema-wise Accuracy ===")
        self.evaluate_by_schema(pred_file)


# Usage example:
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python evaluate.py predictions.json")

    gt_path = Path(__file__).resolve().parent / "ground_truth.json"
    pred_path = Path(sys.argv[1])

    scorer = DBQueryScorer(gt_path)

    scorer.evaluate_all(pred_path)
