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

from structured_outputs import (
    ToolCall as ToolArguments,
    OPERATOR_TO_METHOD,
)

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
# helper ----------------------------------------------------------------------
###############################################################################


def _build_where(ta: ToolArguments):
    """
    Translate ToolArguments filters into a Weaviate Filter tree.
    """
    parts: List[Filter] = []
    # numeric ----------------------------------------------------------------
    if ta.integer_property_filter:
        f = ta.integer_property_filter
        parts.append(
            getattr(Filter.by_property(f.property_name), OPERATOR_TO_METHOD[f.operator])(f.value)
        )
    # text -------------------------------------------------------------------
    if ta.text_property_filter:
        f = ta.text_property_filter
        parts.append(
            getattr(Filter.by_property(f.property_name), OPERATOR_TO_METHOD[f.operator])(f.value)
        )
    # boolean ----------------------------------------------------------------
    if ta.boolean_property_filter:
        f = ta.boolean_property_filter
        parts.append(
            getattr(Filter.by_property(f.property_name), OPERATOR_TO_METHOD[f.operator])(f.value)
        )

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
    USE_CASES = {
        "Restaurants": (
            "This schema focuses on enabling users to discover restaurants based on a comprehensive profile. With semantic search, users can find restaurants by cuisine, ambiance, or special features."
        ),
        "Menus": (
            "This schema assists in linking dining experiences with specific restaurants through their menus. Rich search features allow customers to find dishes tailored to dietary needs and price points."
        ),
        "Reservations": (
            "This schema integrates with the restaurants by managing booking experiences. Semantic search of reservations can uncover trends in dining preferences and commonly requested meal attributes."
        ),
        "Clinics": (
            "This schema aims to help users discover clinics based on services, specialties, and patient satisfaction. Semantic search can be used to find clinics by specific healthcare needs or service qualities."
        ),
        "Doctors": (
            "This schema supports finding doctors based on expertise and experience. With semantic search, users can match their health concerns to the right professionals by exploring detailed profiles."
        ),
        "Appointments": (
            "This schema is designed to manage and optimize booking experiences by allowing semantic searches for specific appointment details and patient booking patterns."
        ),
        "Courses": (
            "This schema helps users find courses based on subject matter, duration, and enrollment status. Semantic search enhances discovery of courses by learning outcomes and topics covered."
        ),
        "Instructors": (
            "This schema allows students and administrators to search for instructors based on experience and background. Rich biographies help in matching students with instructors who align with their learning style and academic goals."
        ),
        "Students": (
            "This schema is designed to help institutions manage student data and preferences. Semantic search allows deeper insights into student research interests and progression paths."
        ),
        "TravelDestinations": (
            "This schema allows users to explore travel destinations based on detailed descriptions and average costs. Semantic search can help users find destinations that match desired experiences or budget levels."
        ),
        "TravelAgents": (
            "This schema supports customers in finding travel agents based on expertise and availability. Semantic search enables matching with agents who have specific regional knowledge or customer service excellence."
        ),
        "TravelPackages": (
            "This schema helps travelers find travel packages based on detailed descriptions and pricing. Semantic search allows for discovering packages that align with preferences for activities or budget constraints."
        ),
        "Museums": (
            "The Museums schema provides an enriching database for those interested in exploring detailed cultural exhibits. Semantic search capabilities highlight unique features and historical value of the museum's collections."
        ),
        "Exhibitions": (
            "This schema helps users discover and explore various exhibitions based on thematic interest or visitor popularity, encouraging semantic searches for immersive cultural experiences."
        ),
        "ArtPieces": (
            "The ArtPieces schema supports the discovery and assessment of art pieces across various museums. With semantic capabilities, users can explore artwork based on historical significance and monetary valuation."
        ),
    }

    def __init__(self):
        self.client = weaviate.connect_to_local(
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

    def __del__(self):
        if self.client:
            self.client.close()
            logger.info("Weaviate connection closed.")

    # ------------------------------------------------------------------ schema
    def create_collection(
        self,
        name: str,
        properties: List[Dict[str, Any]],
        vectorizer: str | None = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        if name in self.client.collections.list_all():
            return

        vectorizer_config = Configure.NamedVectors.text2vec_huggingface(
            name=f"{name.lower()}_vectorizer",
            model=vectorizer,
        )

        props = [
            Property(
                name=p["name"],
                data_type=getattr(DataType, p["data_type"].upper()),
                description=p["description"],
            )
            for p in properties
        ]
        self.client.collections.create(
            name, properties=props, description=self.USE_CASES[name], vectorizer_config=None
        )

    def get_collection_schemas(self, collections):
        """
        Get all collection schemas from the Weaviate client and return a structured string
        with collection name, description, and properties (with descriptions and data types).
        """
        collections_desc = []
        for collection in collections:
            collection_str = ""
            collection_schema = self.client.collections.get(collection)
            config = collection_schema.config.get(simple=True)
            name = config.name
            description = config.description
            collection_str += f"Collection: {name}\n"
            collection_str += f"Description: {description}\n"
            collection_str += "Properties:\n"
            for prop in config.properties:
                prop_name = prop.name
                prop_desc = prop.description
                prop_type = (
                    prop.data_type.value
                    if hasattr(prop.data_type, "value")
                    else str(prop.data_type)
                )
                collection_str += f"  - {prop_name} ({prop_type}): {prop_desc}\n"

            collections_desc.append(collection_str)

        return "\n".join(collections_desc)

    # ------------------------------------------------------------------ insert
    def insert(self, collection_name: str, objs: List[Dict[str, Any]]):
        collection = self.client.collections.get(collection_name)
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
        coll = self.client.collections.get(ta.collection_name)
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
        # second row is data type, read it
        data_type = df.iloc[0][column]
        # third row is the column descriptions, read it
        column_description = df.iloc[1][column]

        type_mapping = {
            "string": "text",
            "number": "int",
            "boolean": "bool",
        }

        properties.append(
            {
                "name": column,
                "data_type": type_mapping.get(data_type, "text"),
                "description": column_description,
            }
        )

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
    collections = db.client.collections.list_all()

    strng = db.get_collection_schemas(collections)
    print(strng)

    exit()

    # Path to directory containing CSV files
    DATA_DIR = os.path.join(os.path.dirname(__file__), "collections_data")

    import json

    with open(os.path.join(DATA_DIR, "collection_descriptions.json"), "r") as f:
        collection_descriptions = json.load(f)

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

        # Read CSV, skip second and third row (as they are not data) and insert data
        df = pd.read_csv(csv_path, skiprows=[1, 2])
        rows = []
        for _, row in df.iterrows():
            # Convert each value to the appropriate type
            converted_row = {
                col: convert_value_by_type(
                    row[col], next((p["data_type"] for p in properties if p["name"] == col), "text")
                )
                for col in row.index
            }
            rows.append(converted_row)

        # Insert data into collection
        if rows:
            db.insert(collection_name, rows)
        else:
            logger.info(f"No rows found in {csv_path}")

    logger.info("CSV data inserted – ready for testing.")

    strng = db.get_collection_schemas(collections)
    print(strng)
