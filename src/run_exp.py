"""
Reades ground_truth.json, for each example, sends the natural language query to the model, save its response in another json file
"""

from weaviate_client import WeaviateDatabase
import llm
import structured_outputs
import eval
import json
import os

db = WeaviateDatabase()
collections = db.client.collections.list_all()
tool_schema = structured_outputs.get_query_tool_schema(collections_list=list(collections.keys()))

predictions_path = "./predictions.json"
ground_truth_path = "/Users/ege/llm-project/src/ground_truth.json"
with open(ground_truth_path, "r") as f:
    ground_truth = json.load(f)

gpt_4o_model = llm.OpenAIClient(model_name="gpt-4o")
llama_model = llm.OllamaClient(model_name="llama3.1:8b")


def run_preds(model, output_path):
    predictions = []
    for i, example in enumerate(ground_truth):
        print(f"Processing example {i + 1}/{len(ground_truth)}")
        query = example["query"]["corresponding_natural_language_query"]

        system_message = (
            "You are a database assistant. You must call the function `query_database` to query a database based on user input.\n\n"
            "The available collections and their descriptions are:\n\n"
            f"{db.get_collection_schemas()}\n\n"
        )

        response = model.generate_response(
            messages=[{"role": "user", "content": query}],
            system_message=system_message,
            tools=[tool_schema],
            # tool_choice="auto",
            temperature=0,
            # tool_choice={"type": "function", "function": {"name": "query_database"}},
        )

        tool_args = response.choices[0].message.tool_calls[0].function.arguments

        try:
            # Parse & validate
            call = structured_outputs.ToolCall(
                function_name=response.choices[0].message.tool_calls[0].function.name,
                arguments=structured_outputs.ToolArguments.model_validate(json.loads(tool_args)),
            )
            is_valid = True
        except Exception:
            is_valid = False

        # add the call to the predictions
        prediction = {
            "query": {
                "corresponding_natural_language_query": query,
                **call.arguments.model_dump(),
            },
            "is_valid": is_valid,
        }

        # In the tool call, the collection name is "collection_name", but in the ground truth file, it is "target_collection"
        prediction["query"]["target_collection"] = prediction["query"].pop("collection_name")

        # add the prediction to the predictions list
        predictions.append(prediction)

        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=4)
        print(f"Saved the {i + 1} predictions to {output_path}")


if __name__ == "__main__":
    # run_preds(gpt_4o_model, predictions_path)
    run_preds(llama_model, predictions_path)
