weaviate_query_api_docs = """
The Database Query Tool provides a flexible interface for querying collections within a database. It supports various operations including full-text search, filtering, aggregations, and grouping. This tool is designed to handle different data types (integer, text, and boolean) with type-specific operations.

Basic Usage
At minimum, each query must specify a collection_name. All other parameters are optional and can be combined to create complex queries.

Core Parameters and Search:
The required parameter is collection_name, which specifies which collection to query. The search_query parameter is ESSENTIAL for finding items based on descriptive terms or phrases - you must use it whenever searching for descriptive qualities of items. Never use text filters (LIKE) for descriptive searches. The groupby_property parameter allows grouping results by a specified property.

CRITICAL: Search vs. Filters
1. ALWAYS use search_query for:
   - Any descriptive terms ("romantic", "cozy", "relaxing")
   - Atmosphere descriptions ("romantic atmosphere", "cozy ambiance")
   - Restaurant types ("brunch spots", "dining locations")
   - Amenities ("outdoor seating")
   - Special characteristics ("vegan-friendly")
   - Combinations of these ("romantic Italian restaurants", "cozy dining spots")

2. NEVER use text filters (LIKE operator) for:
   - Descriptive terms
   - Atmosphere
   - Restaurant types
   - Amenities
   - Special characteristics

3. Use filters ONLY for:
   - Exact numeric comparisons (rating > 4)
   - Exact property matching
   - Boolean conditions

Correct Examples:
✓ search_query: "romantic Italian restaurants"
✓ search_query: "vegan-friendly brunch spots"
✓ search_query: "romantic dining locations"
✓ search_query: "restaurants with relaxing atmosphere"

Incorrect Examples:
✗ text_filter: description LIKE "romantic"
✗ text_filter: description LIKE "vegan"
✗ text_filter: description LIKE "relaxing"
✗ boolean_filter: openNow = True (when calculating percentages)

Key Points About Aggregations:
1. When asked about "how many", use COUNT aggregation
2. When asked about percentages of boolean properties, use PERCENTAGE_TRUE aggregation
3. When asked about "most common" or "typical" text features, use TOP_OCCURRENCES
4. When asked about averages, always include the appropriate MEAN aggregation

Property Usage Rules:
1. For cuisine grouping, use "description.cuisine" as the property
2. For open/closed status:
   - Use boolean_aggregation with PERCENTAGE_TRUE when calculating percentages
   - Use groupby: "openNow" when grouping results
   - Don't use boolean filters unless specifically filtering, not aggregating

Aggregation Operations
The tool provides sophisticated aggregation capabilities for different data types:
Integer Aggregations: Use integer_property_aggregation for numeric analysis. Available metrics: COUNT, TYPE, MIN, MAX, MEAN, MEDIAN, MODE, SUM
Text Aggregations: Use text_property_aggregation for text analysis. Available metrics: COUNT, TYPE, TOP_OCCURRENCES
Boolean Aggregations: Use boolean_property_aggregation for boolean statistics. Available metrics: COUNT, TYPE, TOTAL_TRUE, TOTAL_FALSE, PERCENTAGE_TRUE, PERCENTAGE_FALSE
"""

prompt_general = """
You are a precision-focused Weaviate query generator. Your ONLY task is to output a final Weaviate query that EXACTLY matches the provided schema and NL query. Every element (collection names, property names, filter types, operators, numeric formats, aggregation metrics, and group-by properties) must be an exact match. No substitutions, derivations, or extra text is allowed. Do not reveal any internal reasoning.

Instructions:
1. Analyze the Schema & NL Query:
   • Use ONLY schema values for collection and property names (e.g., "Restaurants", "averageRating").
   • Extract the descriptive search query exactly from the NL query.
THIS IS VERY IMPORTANT!! PLEASE PAY CLOSE ATTENTION TO THIS EXPLANATION OF THE AVAILABLE OPERATORS!!
   • For filters:
       - Text: use Text Filter with LIKE.
       - Numeric: use Integer Filter with operators (=, <, >, <=, >=) and numeric values (include .0 if required).
       - Boolean: use Boolean Filter with "=" and value True.
   • For aggregations, use:
       - Text: TOP_OCCURRENCES.
       - Int: MIN, MAX, MEAN, MEDIAN, MODE, or SUM.
Again, for IntAggregation you have MIN, MAX, MEAN, MEDIAN, MODE, or SUM!!! DO NOT EVER TRY TO USE SOMETHING LIKE TOTAL_TRUE with an IntAggregation!! THIS IS EXTREMELY IMPORTANT!
       - Boolean: TOTAL_TRUE, TOTAL_FALSE, PERCENTAGE_TRUE, or PERCENTAGE_FALSE.
PLEASE REMEMBER THIS!! THIS IS HOW YOU COUNT OBJECTS!!!! DO NOT TRY TO COUNT in Aggregations!! 
   • If counting objects is needed, set total_count to true (do NOT use COUNT in aggregations!!! This is very important!!).
   • Group By must exactly match a schema property.

2. Verification (Internal Only):
   • Confirm every element exactly matches the schema—no extra filters or modifications.

3. Output Format (Output ONLY):
Weaviate Query Details:
  Target Collection: <exact collection>
  Search Query: <exact search text>
  Total Count: <true/false>
  [Filters if any:]
    • Filter Type: <Text Filter / Integer Filter / Boolean Filter>
    • Property: <exact property name>
    • Operator: <exact operator>
    • Value: <exact value>
  [Aggregations if any:]
    • Aggregation Type: <Text / Boolean / Integer Aggregation>
    • Property: <exact property name>
    • Metrics: <exact metric>
  Group By: <if applicable, exact property name>
  Natural Language Query: {query}

User Query:
{query}

Available Schema (Collections and Properties):
{collection_infos}

Now, generate the final Weaviate query following these guidelines.
IMPORTANT!! Please remember, COUNT and TYPE are not valid aggregations for an IntAggregation, TextAggregation, or BooleanAggregation!
IMPORTANT!! Please remember to format your response as a function call with the arguments you have chosen.
IMPORTANT!! IT IS VERY COMMON TO OUTPUT INCORRECT `IntAggregation` METRICS! PLEASE NOTE!! For IntAggregation,
Input should be 'MIN', 'MAX', 'MEAN', 'MEDIAN', 'MODE' or 'SUM', otherwise you will get an error such as: [type=literal_error, input_value='COUNT', input_type=str]
THIS IS EXTERMELY IMPORTANT! YOUR NUMBER 1 FOCUS SHOULD BE TO MAKE SURE THESE QUERIES ARE CORRECTLY FORMATTED!!!
REMEMBER, DO NOT EVERY TRY TO USE, say COUNT, with an IntAggregation!! YOU COUNT WITH THE `total_count` ARGUMENT!!! IntAggregation only supporst MIN, MAX, MEAN, MEDIAN, MODE, or SUM!!
"""

preference_ranking_prompt = """
You are an expert at evaluating database query predictions.

Given this natural language query:

{natural_language_query}

We have the following model predictions:

{predictions_text}

You must produce valid JSON with:

A top-level field named "rationale" (a string explanation of your ranking decisions)
One integer field for each LLM name, with no extra fields.
For example:
{{
"rationale": "some explanation",
{llm_fields_str}
}}

Where:

"rationale" is a string
Each LLM name is a required integer rank (1 = best, 2 = next best, etc.)
No other keys are allowed
All discovered LLMs must appear exactly once
Provide unique ranks (no ties).
IMPORTANT: Respond ONLY with the JSON object, no additional text or formatting.
"""
