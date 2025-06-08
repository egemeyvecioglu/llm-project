"""
Reades ground_truth.json, for each example, sends the natural language query to the model, save its response in another json file
"""

from weaviate_client import WeaviateDatabase
import llm
import structured_outputs
import eval
import json
import os

from prompts import weaviate_query_api_docs, prompt_general


def run_preds(model, output_path):
    predictions = []
    for i, example in enumerate(ground_truth):
        # import time

        # time.sleep(60)
        print(f"Processing example {i + 1}/{len(ground_truth)}")
        query = example["query"]["corresponding_natural_language_query"]

        collections_for_the_example = [
            collection["name"]
            for collection in json.loads(example["database_schema"])["weaviate_collections"]
        ]

        collection_infos = db.get_collection_schemas(collections_for_the_example)

        instructions = (
            weaviate_query_api_docs
            + "\n\n"
            + prompt_general.format(query=query, collection_infos=collection_infos)
        )

        # instructions += """

        # Reminder:
        # - Function calls MUST follow the exact specified format
        # - The field names MUST match exactly as specified - use "property_name" not "property", "metrics" not "function", etc. Follow the required format for all objects.
        # - Required parameters MUST be specified

        # """

        tool_schema = structured_outputs.get_query_tool_schema(
            collections_list=collections_for_the_example
        )

        response = model.generate_response(
            messages=[{"role": "user", "content": instructions}],
            tools=[tool_schema],
            system_message="You are a helpful assistant. Use the supplied tools to assist the user.",
            temperature=0,
        )

        # try to get tool call response field
        tool_name, tool_args = model.get_tool_call_response(response)
        # if not available, check the content field for a json code block including tool call
        if not tool_args:
            tool_name = "query_database"
            tool_params = model.get_message_response(response)
            try:
                tool_params = tool_params[tool_params.find("{") : tool_params.rfind("}") + 1]
                tool_params = (
                    tool_params.replace("True", "true")
                    .replace("False", "false")
                    .replace("None", "null")
                )
            # if also content field does not include a valid JSON, set to empty dict
            except (ValueError, json.JSONDecodeError):
                tool_name = ""
                tool_args = "{}"

            tool_called = False

            if isinstance(tool_params, str):
                try:
                    tool_json = json.loads(tool_params)
                except Exception as e:
                    print("Error parsing JSON from text response: ", e)
                    tool_json = {}

            tool_name = tool_json.get("name", "")
            tool_args = tool_json.get("parameters", {})
        else:
            if not isinstance(tool_args, dict):
                tool_args = json.loads(tool_args)
            tool_called = True

        try:
            # Parse & validate
            call = structured_outputs.ToolCall(
                function_name=tool_name,
                arguments=structured_outputs.ToolArguments.model_validate(tool_args),
            )
            is_valid = True

        except Exception as e:
            print(f"Validation error: {e}")
            is_valid = False

        prediction = {
            "query": {
                "corresponding_natural_language_query": query,
                **tool_args,
            },
            "is_valid": is_valid,
            "tool_called": tool_called,
        }
        # In the tool call, the collection name is "collection_name", but in the ground truth file, it is "target_collection"
        if "collection_name" in prediction["query"]:
            prediction["query"]["target_collection"] = prediction["query"].pop("collection_name")
        else:
            prediction["query"]["target_collection"] = None

        # add the prediction to the predictions list
        predictions.append(prediction)

        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=4)
        print(f"Saved the {i + 1} predictions to {output_path}")


if __name__ == "__main__":
    db = WeaviateDatabase()
    collections = db.client.collections.list_all()

    ground_truth_path = "/path/to/llm-project/src/ground_truth.json"
    with open(ground_truth_path, "r") as f:
        ground_truth = json.load(f)

    # gpt_4o_model = llm.AzureOpenAIClient(model_name="bopti-gpt-4o")
    # gpt_4_1_model = llm.AzureOpenAIClient(model_name="bopti-gpt-4-1")
    # gpt_4o_mini_model = llm.OpenAIClient(model_name="gpt-4o-mini")
    # llama_model = llm.OllamaClient(model_name="llama3.1:8b")
    # claude_model = llm.AnthropicClient(model_name="claude-3-5-sonnet-20240620")
    # gemini_flash_model = llm.GeminiClient("gemini-2.0-flash")

    # run_preds(gpt_4_1_model, "gpt_4-1_predictions.json")
    # run_preds(gpt_4o_model, "./gpt_4o_predictions.json")
    # run_preds(llama_model, "./llama_predictions_new23.json")
    # run_preds(claude_model, "./claude_predictions3.json")
    # run_preds(gemini_flash_model, "./gemini_flash_predictions.json")

    db.client.close()
