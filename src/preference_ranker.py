import argparse, json, csv, time
from pathlib import Path
from prompts import preference_ranking_prompt

from llm import AzureOpenAIClient

MODEL_NAME = "YOUR_MODEL_NAME"
# Wait 3 s between judge calls so we don’t hammer the endpoint
DELAY_BETWEEN_CALLS = 3.0


class PreferenceRanker:

    def __init__(self, model_name: str = MODEL_NAME, delay: float = DELAY_BETWEEN_CALLS):
        self.model_name = model_name
        self.delay = delay
        self.judge = AzureOpenAIClient(model_name=self.model_name)

    @staticmethod
    def load_predictions(paths: list[str]) -> dict[str, list[dict]]:
        """Map model-name → list[per-example-dict]."""
        out: dict[str, list[dict]] = {}
        for p in paths:
            with open(p, "r") as f:
                recs = json.load(f)
            model_name = Path(p).stem.replace("_predictions", "")
            out[model_name] = recs
        # basic sanity check – all lists must be same length & same NL order
        lens = {len(v) for v in out.values()}
        if len(lens) != 1:
            raise ValueError(f"Prediction files have differing lengths: {lens}")
        return out

    def run_ranking(self, pred_paths: list[str], ground_truth_path: str, out_csv: str):
        gt = json.load(open(ground_truth_path))
        preds = self.load_predictions(pred_paths)
        model_names = list(preds.keys())
        n_examples = len(next(iter(preds.values())))

        # Create the llm_fields_str for the example JSON format
        llm_fields_example = ",\n".join(f'"{name}": <rank>' for name in model_names)
        print(llm_fields_example)
        input("Press enter to continue")

        fieldnames = ["example_idx", "ranking_json", "rationale"] + [
            f"rank_{m}" for m in model_names
        ]
        with open(out_csv, "w", newline="") as csv_f:
            writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(n_examples):
                nl_query = gt[i]["query"]["corresponding_natural_language_query"]

                # Build predictions text
                predictions_blocks = []
                for m in model_names:
                    args = json.dumps(preds[m][i]["query"], ensure_ascii=False)
                    predictions_blocks.append(f"### {m}\n{args}")

                predictions_text = "\n\n".join(predictions_blocks)

                # Format the prompt
                user_prompt = preference_ranking_prompt.format(
                    natural_language_query=nl_query,
                    predictions_text=predictions_text,
                    llm_fields_str=llm_fields_example,
                )

                response = self.judge.generate_response(
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0,
                )

                raw = self.judge.get_message_response(response)
                try:
                    ranking_payload = json.loads(raw)
                except json.JSONDecodeError:
                    raise ValueError(
                        f"Judging LLM did not return valid JSON on example {i}:\n{raw}"
                    )

                # Convert numerical rankings back to ordered list
                ranking = sorted(model_names, key=lambda m: ranking_payload[m])

                # Store CSV row
                row = {
                    "example_idx": i,
                    "ranking_json": json.dumps(ranking),
                    "rationale": ranking_payload["rationale"],
                }
                for model in model_names:
                    row[f"rank_{model}"] = ranking_payload[model]
                writer.writerow(row)
                csv_f.flush()

                print(f"[{i+1:03}/{n_examples}] Judge ranks: {ranking}")
                time.sleep(self.delay)

        print(f"\nPreference ranks written to {out_csv}")

    def compute_stats(self, csv_path: str):
        weights = {1: 100, 2: 70, 3: 50, 4: 35, 5: 25, 6: 20, 7: 15, 8: 10}
        counts = {}
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            # collect model names from header
            model_names = [
                k.removeprefix("rank_") for k in reader.fieldnames if k.startswith("rank_")
            ]
            for m in model_names:
                counts[m] = {"top1": 0, "weighted": 0}
            for row in reader:
                for model in model_names:
                    rank = int(row[f"rank_{model}"])
                    counts[model]["weighted"] += weights.get(rank, 0)
                    if rank == 1:
                        counts[model]["top1"] += 1
        print("Model statistics:")
        for model, stats in counts.items():
            print(f"{model}: Top1={stats['top1']} | WeightedScore={stats['weighted']}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="LLM-as-Judge preference ranker")
    parser.add_argument(
        "--pred-dir",
        required=True,
        help="Directory containing *_predictions.json files",
    )
    parser.add_argument("--out", default="judge_ranks.csv", help="Output CSV path")
    args = parser.parse_args()
    pred_paths = list(Path(args.pred_dir).glob("*_predictions.json"))
    pr = PreferenceRanker()
    pr.run_ranking(
        pred_paths=pred_paths,
        ground_truth_path=str(Path(__file__).parent / "ground_truth.json"),
        out_csv=args.out,
    )
    pr.compute_stats(args.out)
