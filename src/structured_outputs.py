from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import json
from llm import OpenAIClient


def get_query_tool_schema(collections_list):
    query_database_tool = {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": (
                "Query a database with an optional search query or optional filters or aggregations on the results.\n\nIMPORTANT! Please be mindful of the available query APIs you can use such as search queries, filters, aggregations, and groupby\n\nAvailable collections in this database:\n{ collections_description }"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "The collection to query.",
                        "enum": collections_list,
                    },
                    "search_query": {
                        "type": "string",
                        "description": "A search query to return objects from a search index.",
                    },
                    "integer_property_filter": {
                        "type": "object",
                        "description": "Filter numeric properties using comparison operators.",
                        "properties": {
                            "property_name": {"type": "string"},
                            "operator": {"type": "string", "enum": ["=", "<", ">", "<=", ">="]},
                            "value": {"type": "number"},
                        },
                    },
                    "text_property_filter": {
                        "type": "object",
                        "description": "Filter text properties using equality or LIKE operators",
                        "properties": {
                            "property_name": {"type": "string"},
                            "operator": {"type": "string", "enum": ["=", "LIKE"]},
                            "value": {"type": "string"},
                        },
                    },
                    "boolean_property_filter": {
                        "type": "object",
                        "description": "Filter boolean properties using equality operators",
                        "properties": {
                            "property_name": {"type": "string"},
                            "operator": {"type": "string", "enum": ["=", "!="]},
                            "value": {"type": "boolean"},
                        },
                    },
                    "integer_property_aggregation": {
                        "type": "object",
                        "description": "Aggregate numeric properties using statistical functions",
                        "properties": {
                            "property_name": {"type": "string"},
                            "metrics": {
                                "type": "string",
                                "enum": [
                                    "COUNT",
                                    "TYPE",
                                    "MIN",
                                    "MAX",
                                    "MEAN",
                                    "MEDIAN",
                                    "MODE",
                                    "SUM",
                                ],
                            },
                        },
                    },
                    "text_property_aggregation": {
                        "type": "object",
                        "description": "Aggregate text properties using frequency analysis",
                        "properties": {
                            "property_name": {"type": "string"},
                            "metrics": {
                                "type": "string",
                                "enum": ["COUNT", "TYPE", "TOP_OCCURRENCES"],
                            },
                            "top_occurrences_limit": {"type": "integer"},
                        },
                    },
                    "boolean_property_aggregation": {
                        "type": "object",
                        "description": "Aggregate boolean properties using statistical functions",
                        "properties": {
                            "property_name": {"type": "string"},
                            "metrics": {
                                "type": "string",
                                "enum": [
                                    "COUNT",
                                    "TYPE",
                                    "TOTAL_TRUE",
                                    "TOTAL_FALSE",
                                    "PERCENTAGE_TRUE",
                                    "PERCENTAGE_FALSE",
                                ],
                            },
                        },
                    },
                    "groupby_property": {
                        "type": "string",
                        "description": "Group the results by a property.",
                    },
                },
                "required": ["collection_name"],
            },
        },
    }

    return query_database_tool


# ──────────────────────────────────────────────────────────────────────────────
# enums
# ──────────────────────────────────────────────────────────────────────────────
class Operator(str, Enum):
    EQ = "="
    NEQ = "!="
    LT = "<"
    LTE = "<="
    GT = ">"
    GTE = ">="
    LIKE = "LIKE"


class IntMetric(str, Enum):
    COUNT = "COUNT"
    TYPE = "TYPE"
    MIN = "MIN"
    MAX = "MAX"
    MEAN = "MEAN"
    MEDIAN = "MEDIAN"
    MODE = "MODE"
    SUM = "SUM"


class TextMetric(str, Enum):
    COUNT = "COUNT"
    TOP_OCCURRENCES = "TOP_OCCURRENCES"
    TYPE = "TYPE"


class BoolMetric(str, Enum):
    COUNT = "COUNT"
    TYPE = "TYPE"
    TOTAL_TRUE = "TOTAL_TRUE"
    TOTAL_FALSE = "TOTAL_FALSE"
    PERCENTAGE_TRUE = "PERCENTAGE_TRUE"
    PERCENTAGE_FALSE = "PERCENTAGE_FALSE"


OPERATOR_TO_METHOD: Dict[Operator, str] = {
    Operator.EQ: "equal",
    Operator.NEQ: "not_equal",
    Operator.LT: "less_than",
    Operator.LTE: "less_or_equal",
    Operator.GT: "greater_than",
    Operator.GTE: "greater_or_equal",
    Operator.LIKE: "like",
}


# ──────────────────────────────────────────────────────────────────────────────
# sub‑objects
# ──────────────────────────────────────────────────────────────────────────────
class IntPropertyFilter(BaseModel):
    property_name: str
    operator: Operator
    value: int


class TextPropertyFilter(BaseModel):
    property_name: str
    operator: Operator
    value: str


class BooleanPropertyFilter(BaseModel):
    property_name: str
    operator: Operator
    value: bool


class IntAggregation(BaseModel):
    property_name: str
    metrics: IntMetric


class TextAggregation(BaseModel):
    property_name: str
    metrics: TextMetric
    top_occurrences_limit: Optional[int] = Field(default=None, ge=1)


class BooleanAggregation(BaseModel):
    property_name: str
    metrics: BoolMetric


class ToolArguments(BaseModel):
    collection_name: str
    search_query: Optional[str] = None
    integer_property_filter: Optional[IntPropertyFilter] = None
    text_property_filter: Optional[TextPropertyFilter] = None
    boolean_property_filter: Optional[BooleanPropertyFilter] = None
    integer_property_aggregation: Optional[IntAggregation] = None
    text_property_aggregation: Optional[TextAggregation] = None
    boolean_property_aggregation: Optional[BooleanAggregation] = None
    groupby_property: Optional[str] = None


class ToolCall(BaseModel):
    function_name: str
    arguments: ToolArguments


class ResponseOfToolCall(BaseModel):
    tool_rationale: Optional[str] = Field(
        default=None, description="A rationale regarding whether tool calls are needed."
    )
    use_tools: bool
    response: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


OPERATOR_TO_METHOD: Dict[Operator, str] = {
    Operator.EQ: "equal",
    Operator.NEQ: "not_equal",
    Operator.LT: "less_than",
    Operator.LTE: "less_or_equal",
    Operator.GT: "greater_than",
    Operator.GTE: "greater_or_equal",
    Operator.LIKE: "like",
}

# ──────────────────────────────────────────────────────────────────────────────
# self‑test / demos
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from openai import OpenAI
    from weaviate_client import WeaviateDatabase

    db = WeaviateDatabase()

    # Get collections list from Weaviate
    collections = db.client.collections.list_all()
    tool_schema = get_query_tool_schema(collections_list=list(collections.keys()))

    def local_demo() -> None:
        call = ToolCall(
            function_name="query_database",
            arguments=ToolArguments(
                collection_name="restaurants",
                search_query="cozy ambiance",
                integer_property_filter=IntPropertyFilter(
                    property_name="average_rating",
                    operator=Operator.GTE,
                    value=4,
                ),
                text_property_filter=TextPropertyFilter(
                    property_name="cuisine",
                    operator=Operator.LIKE,
                    value="Italian",
                ),
                boolean_property_aggregation=BooleanAggregation(
                    property_name="is_open_now",
                    metrics=BoolMetric.COUNT,
                ),
            ),
        )
        print("Local ToolCall args:", json.dumps(call.model_dump(), indent=2))

    def openai_demo() -> None:
        # Minimal system prompt that exposes only schema + function signature
        system = (
            "You are a database assistant. You must call the function `query_database` to query a database based on user input.\n\n"
            "The available collections and their descriptions are:\n\n"
            f"{db.get_collection_schemas()}\n\n"
        )

        user = {
            "role": "user",
            "content": (
                "Find restaurants with a cozy ambiance and Italian cuisine, where the average rating is at least 4, count how many such restaurants there are, and group them by whether they are currently open or not."
            ),
        }

        client = OpenAIClient(model_name="gpt-4o-mini")
        response = client.generate_response(
            messages=[user],
            system_message=system,
            tools=[tool_schema],
            # tool_choice={"type": "function", "function": {"name": "query_database"}},
            tool_choice="auto",
        )

        tool_args = response.choices[0].message.tool_calls[0].function.arguments
        print("GPT-generated args:", tool_args)

        # Parse & validate
        call = ToolCall(
            function_name=response.choices[0].message.tool_calls[0].function.name,
            arguments=ToolArguments.model_validate(json.loads(tool_args)),
        )
        print("Parsed ToolCall:", json.dumps(call.model_dump(), indent=2))

    # Run demos
    local_demo()
    print("\n" + "=" * 80 + "\n")
    openai_demo()

    db.client.close()
