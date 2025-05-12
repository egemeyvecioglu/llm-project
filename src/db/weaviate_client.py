from __future__ import annotations
import os
import logging
import pandas as pd
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter
from pydantic import BaseModel
import glob

import weaviate.util

###############################################################################
# env -------------------------------------------------------------------------
###############################################################################
load_dotenv()
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_HTTP_PORT", 8080))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", 50051))
HUGGINGFACE_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
###############################################################################

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


###############################################################################
# tool‑call schema (pydantic) --------------------------------------------------
###############################################################################


class _NumericFilter(BaseModel):
    property_name: str
    operator: str  # = < <= > >=
    value: float


class _TextFilter(BaseModel):
    property_name: str
    operator: str  # = LIKE
    value: str


class _BoolFilter(BaseModel):
    property_name: str
    operator: str  # = !=
    value: bool


class _NumericAgg(BaseModel):
    property_name: str
    metrics: str  # COUNT SUM MIN MAX MEAN


class _TextAgg(BaseModel):
    property_name: str
    metrics: str  # COUNT TOP_OCCURRENCES TYPE
    top_occurrences_limit: Optional[int] = None


class _BoolAgg(BaseModel):
    property_name: str
    metrics: str  # COUNT TOTAL_TRUE PERCENTAGE_TRUE


class ToolArguments(BaseModel):
    collection_name: str
    search_query: Optional[str] = None
    integer_property_filter: Optional[_NumericFilter] = None
    text_property_filter: Optional[_TextFilter] = None
    boolean_property_filter: Optional[_BoolFilter] = None
    integer_property_aggregation: Optional[_NumericAgg] = None
    text_property_aggregation: Optional[_TextAgg] = None
    boolean_property_aggregation: Optional[_BoolAgg] = None
    groupby_property: Optional[str] = None


###############################################################################
# helper ----------------------------------------------------------------------
###############################################################################

_OP = {
    "=": "equal",
    "!=": "not_equal",
    "<": "less_than",
    "<=": "less_or_equal",
    ">": "greater_than",
    ">=": "greater_or_equal",
    "LIKE": "like",
}


def _build_where(ta: ToolArguments):
    parts: List[Filter] = []
    if ta.integer_property_filter:
        f = ta.integer_property_filter
        parts.append(getattr(Filter.by_property(f.property_name), _OP[f.operator])(f.value))
    if ta.text_property_filter:
        f = ta.text_property_filter
        parts.append(getattr(Filter.by_property(f.property_name), _OP[f.operator])(f.value))
    if ta.boolean_property_filter:
        f = ta.boolean_property_filter
        parts.append(getattr(Filter.by_property(f.property_name), _OP[f.operator])(f.value))
    if not parts:
        return None
    node = parts[0]
    for p in parts[1:]:
        node = node & p
    return node


###############################################################################
# main wrapper ----------------------------------------------------------------
###############################################################################


class WeaviateDatabase:
    def __init__(self):
        self._client = weaviate.connect_to_local(
            host=WEAVIATE_HOST,
            port=WEAVIATE_HTTP_PORT,
            grpc_port=WEAVIATE_GRPC_PORT,
            headers=(
                {"X-HuggingFace-Api-Key": HUGGINGFACE_ACCESS_TOKEN}
                if HUGGINGFACE_ACCESS_TOKEN
                else None
            ),
            additional_config=weaviate.classes.init.AdditionalConfig(
                timeout=weaviate.classes.init.Timeout(init=60, query=60, insert=180)
            ),
        )
        logger.info("Connected to Weaviate at %s:%s", WEAVIATE_HOST, WEAVIATE_HTTP_PORT)

    # ------------------------------------------------------------------ schema
    def create_collection(
        self,
        name: str,
        properties: List[Dict[str, Any]],
        vectorizer: str | None = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        if name in self._client.collections.list_all():
            return

        vectorizer_config = Configure.NamedVectors.text2vec_huggingface(
            name=f"{name.lower()}_vectorizer",
            model=vectorizer,
        )

        props = [
            Property(name=p["name"], data_type=getattr(DataType, p["data_type"].upper()))
            for p in properties
        ]
        self._client.collections.create(name, properties=props, vectorizer_config=None)

    # ------------------------------------------------------------------ insert
    def insert(self, collection_name: str, objs: List[Dict[str, Any]]):
        collection = self._client.collections.get(collection_name)
        try:
            with collection.batch.dynamic() as batch:
                for i, data_row in enumerate(objs):
                    uuid = weaviate.util.generate_uuid5(data_row)
                    batch.add_object(properties=data_row, uuid=uuid)
                    if batch.number_errors > 10:
                        print("Batch import stopped due to excessive errors.")
                        break

            failed_objects = collection.batch.failed_objects
            if failed_objects:
                print(f"Number of failed imports: {len(failed_objects)}")
                print(f"First failed object: {failed_objects[0]}")
        except Exception as e:
            print(f"Error during batch import: {e}")
            raise
        logger.info("Inserted %d objects into %s", len(objs), collection_name)

    # ------------------------------------------------------------------ query
    def query(self, **kwargs):
        ta = ToolArguments(**kwargs)
        coll = self._client.collections.get(ta.collection_name)
        where = _build_where(ta)
        if (
            ta.integer_property_aggregation
            or ta.text_property_aggregation
            or ta.boolean_property_aggregation
        ):
            agg = coll.aggregate
            if where:
                agg = agg.with_where(where)
            if ta.groupby_property:
                agg = agg.with_group_by(ta.groupby_property)
            return agg.do()
        q = coll.query
        if ta.search_query:
            q = q.near_text(query=ta.search_query, limit=20)
        if where:
            q = q.with_where(where)
        if ta.groupby_property:
            q = q.with_group_by(ta.groupby_property)
        return q.do()


###############################################################################
# quick‑start demo ------------------------------------------------------------
###############################################################################


def infer_data_type(series: pd.Series) -> str:
    """
    Infer the appropriate data type for a column based on its content.

    Args:
        series: pandas Series containing column data

    Returns:
        String representation of the inferred data type ('text', 'number', or 'bool')
    """
    # Check if column contains only boolean values
    if (
        series.dropna()
        .map(lambda x: str(x).lower() in ("true", "false", "1", "0", "yes", "no"))
        .all()
    ):
        return "bool"

    # Check if column can be converted to numeric
    try:
        pd.to_numeric(series, errors="raise")
        return "number"
    except (ValueError, TypeError):
        pass

    # Default to text
    return "text"


def get_schema_from_csv(csv_path: str) -> List[Dict[str, str]]:
    """
    Extract schema from a CSV file by analyzing its columns and data types.

    Args:
        csv_path: Path to the CSV file

    Returns:
        List of property definitions with name and data_type
    """
    # Read CSV into pandas DataFrame
    df = pd.read_csv(csv_path)

    # Generate schema properties
    properties = []
    for column in df.columns:
        data_type = infer_data_type(df[column])
        properties.append({"name": column, "data_type": data_type})

    return properties


def extract_collection_name_from_path(csv_path: str) -> str:
    """
    Extract collection name from CSV file path.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Collection name derived from the filename
    """
    # Get the base filename without extension
    basename = os.path.basename(csv_path)
    collection_name = os.path.splitext(basename)[0]

    # Convert first letter to uppercase for consistency
    return collection_name[0].upper() + collection_name[1:]


def convert_value_by_type(value: str, data_type: str) -> Any:
    """
    Convert a string value to the appropriate Python type based on the data type.

    Args:
        value: String value to convert
        data_type: Target data type ('text', 'number', or 'bool')

    Returns:
        Converted value
    """
    if value is None or value == "":
        return None

    if data_type == "number":
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    elif data_type == "bool":
        return str(value).lower() in ("true", "1", "yes")
    else:
        return value


if __name__ == "__main__":
    db = WeaviateDatabase()

    # Path to directory containing CSV files
    DATA_DIR = os.path.join(os.path.dirname(__file__), "exp_data")

    # Find all CSV files
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files in {DATA_DIR}")

    # Dictionary to store schemas
    schemas = {}

    # Process each CSV file
    for csv_path in csv_files:
        # Extract collection name from filename
        collection_name = extract_collection_name_from_path(csv_path)

        # Generate schema from CSV
        properties = get_schema_from_csv(csv_path)

        # Store schema
        schemas[collection_name] = properties

        # Create collection
        logger.info(f"Creating collection {collection_name} with {len(properties)} properties")
        db.create_collection(collection_name, properties=properties)

    # Insert data from CSV files
    for collection_name, properties in schemas.items():
        csv_path = os.path.join(DATA_DIR, f"{collection_name.lower()}.csv")

        if not os.path.exists(csv_path):
            logger.warning(f"CSV file not found for collection {collection_name}: {csv_path}")
            continue

        # Read data using pandas for better type handling
        df = pd.read_csv(csv_path)

        # Convert DataFrame to list of dictionaries
        rows = []
        for _, row in df.iterrows():
            converted = {}
            for prop in properties:
                name = prop["name"]
                data_type = prop["data_type"]
                val = row.get(name)
                converted[name] = convert_value_by_type(val, data_type)
            rows.append(converted)

        # Insert data into collection
        if rows:
            db.insert(collection_name, rows)
        else:
            logger.info(f"No rows found in {csv_path}")

    logger.info("CSV data inserted – ready for testing.")
