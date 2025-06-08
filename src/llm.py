import time
import copy
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from openai import OpenAI, AzureOpenAI
from anthropic import Anthropic, _exceptions
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv("../.env")

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class LLM:
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the LLM with a model name and any additional parameters.
        Args:
            model_name (str): The name of the model to use.
            **kwargs: Additional keyword arguments to pass to the model's constructor. Can be anything that model's api or architecture supports.
        """
        self.model_name = model_name
        self.client = self._create_client(**kwargs)

    def _create_client(self, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_tool_call_response(self, response):
        """
        Processes and returns the response from a tool call invocation.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def get_message_response(self, response):
        """
        Returns message (str) response from raw LLM output.
        """

        raise NotImplementedError("Subclasses should implement this method.")

    def generate_response(self, messages: list, system_message: str = None, **kwargs):
        """
        Generate a response from the model based on the provided messages and system message.

        Args:
            messages (list): A list of messages to send to the model.
            system_message (str, optional): A system message to provide context for the model.
            **kwargs: Additional keyword arguments to pass to the model's generate method. It can be anything that model's api or architecture supports.

        Returns:
            response: The generated response from the model.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class OpenAIClient(LLM):
    def _create_client(self, **kwargs):
        return OpenAI(api_key=OPENAI_API_KEY, **kwargs)

    def get_tool_call_response(self, response):
        try:
            tool_call = response.choices[0].message.tool_calls[0]
            return tool_call.function.name, tool_call.function.arguments
        except:
            return None, None

    def get_message_response(self, response):
        try:
            return response.choices[0].message.content
        except:
            return None

    def generate_response(self, messages: list, system_message: str = None, **kwargs):
        if system_message:
            messages.insert(0, {"role": "system", "content": system_message})

        response = self.client.chat.completions.create(
            model=self.model_name, messages=messages, **kwargs
        )

        return response


class AzureOpenAIClient(OpenAIClient):
    def _create_client(self, **kwargs):
        return AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            **kwargs,
        )


import ollama
from typing import Dict, List, Optional, Tuple


class OllamaClient(LLM):
    def _create_client(self, **kwargs):
        # Ollama doesn't require explicit client initialization like OpenAI
        # We can store any connection parameters if needed
        self.host = kwargs.get("host", "http://localhost:11434")
        return None  # ollama module is used directly

    def get_tool_call_response(self, response):
        try:
            # Check if the response has tool_calls (ChatResponse object)
            if (
                hasattr(response, "message")
                and hasattr(response.message, "tool_calls")
                and response.message.tool_calls
            ):
                tool_call = response.message.tool_calls[0]
                return tool_call.function.name, tool_call.function.arguments
            return None, None
        except:
            return None, None

    def get_message_response(self, response):
        try:
            # Handle ChatResponse object format
            if hasattr(response, "message") and hasattr(response.message, "content"):
                return response.message.content
            return None
        except:
            return None

    def generate_response(self, messages: list, system_message: str = None, **kwargs):
        if system_message:
            messages.insert(0, {"role": "system", "content": system_message})

        # Extract tools from kwargs if provided
        tools = kwargs.pop("tools", None)

        # Separate Ollama-specific parameters from options
        ollama_params = ["model", "messages", "tools", "format", "options", "stream", "keep_alive"]
        options_params = [
            "temperature",
            "top_p",
            "top_k",
            "repeat_penalty",
            "seed",
            "num_predict",
            "num_ctx",
        ]

        # Build options dict for sampling parameters
        options = {}
        for param in list(kwargs.keys()):
            if param in options_params:
                options[param] = kwargs.pop(param)

        # Prepare the request parameters
        chat_params = {
            "model": self.model_name,
            "messages": messages,
        }

        # Add options if any sampling parameters were provided
        if options:
            chat_params["options"] = options

        # Add tools if provided
        if tools:
            chat_params["tools"] = tools

        # Add any remaining valid Ollama parameters
        for param, value in kwargs.items():
            if param in ollama_params:
                chat_params[param] = value

        # Make the request to Ollama
        response = ollama.chat(**chat_params)

        return response


class OllamaOpenAIClient(OpenAIClient):
    """
    Client that uses OpenAI's API format but connects to a local Ollama instance.
    """

    def _create_client(self, host="http://localhost:11434/v1", **kwargs):
        # Configure the OpenAI client to use the local Ollama endpoint
        return OpenAI(
            base_url=host,
            api_key=OLLAMA_API_KEY,  # This is required but the value doesn't matter for Ollama
            **kwargs,
        )


class AnthropicClient(LLM):
    def _create_client(self, **kwargs):
        return Anthropic(api_key=ANTHROPIC_API_KEY, **kwargs)

    def get_tool_call_response(self, response):
        try:
            for content_block in response.content:
                if content_block.type == "tool_use":
                    return content_block.name, content_block.input
            return None, None
        except:
            return None, None

    def get_message_response(self, response):
        try:
            text_blocks = [block.text for block in response.content if block.type == "text"]
            return "".join(text_blocks) if text_blocks else None
        except:
            return None

    def generate_response(self, messages: list, system_message: str = None, **kwargs):
        if kwargs.get("tools"):
            # Completely rebuild the tools structure to ensure no extra fields
            tools = kwargs.pop("tools")
            anthropic_tools = []
            for tool in tools:
                if tool.get("type") == "function":
                    anthropic_tool = copy.deepcopy(tool["function"])
                    anthropic_tool["input_schema"] = anthropic_tool["parameters"]
                    del anthropic_tool["parameters"]

                    # Add the tool back to the list
                    anthropic_tools.append(anthropic_tool)

            kwargs["tools"] = anthropic_tools

        if system_message:
            kwargs["system"] = system_message

        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=messages,
                max_tokens=8192,
                **kwargs,
            )
        except _exceptions.OverloadedError as e:
            time.sleep(30)
            response = self.client.messages.create(
                model=self.model_name,
                messages=messages,
                max_tokens=8192,
                **kwargs,
            )

        return response


class GeminiClient(LLM):
    def _create_client(self, **kwargs):
        genai.configure(api_key=GEMINI_API_KEY)
        return genai.GenerativeModel(self.model_name, **kwargs)

    def get_tool_call_response(self, response):
        try:
            for part in response.parts:
                if hasattr(part, "function_call") and part.function_call:
                    # Convert MapComposite to regular dict
                    args_dict = {}
                    for key, value in part.function_call.args.items():
                        # Handle nested MapComposite objects
                        if hasattr(value, "items"):  # It's a MapComposite
                            args_dict[key] = dict(value)
                        else:
                            args_dict[key] = value
                    return part.function_call.name, args_dict
            return None, None
        except:
            return None, None

    def get_message_response(self, response):
        try:
            return response.text
        except:
            return None

    def generate_response(self, messages: list, system_message: str = None, **kwargs):
        # Convert messages to Gemini format
        gemini_messages = []

        for msg in messages:
            if msg["role"] == "user":
                gemini_messages.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                gemini_messages.append({"role": "model", "parts": [msg["content"]]})

        # Handle tools if provided
        tools = kwargs.pop("tools", None)
        gemini_tools = None
        if tools:
            gemini_tools = []
            for tool in tools:
                if tool.get("type") == "function":
                    func_def = tool["function"]

                    # Convert OpenAI schema to Gemini schema format
                    gemini_schema = self._convert_schema_to_gemini(func_def["parameters"])

                    gemini_tool = genai.protos.Tool(
                        function_declarations=[
                            genai.protos.FunctionDeclaration(
                                name=func_def["name"],
                                description=func_def.get("description", ""),
                                parameters=gemini_schema,
                            )
                        ]
                    )
                    gemini_tools.append(gemini_tool)

        # Handle generation config - only keep supported parameters
        generation_config = {}
        if "temperature" in kwargs:
            generation_config["temperature"] = kwargs.pop("temperature")
        if "max_tokens" in kwargs:
            generation_config["max_output_tokens"] = kwargs.pop("max_tokens")
        if "top_p" in kwargs:
            generation_config["top_p"] = kwargs.pop("top_p")
        if "top_k" in kwargs:
            generation_config["top_k"] = kwargs.pop("top_k")

        # Create generation config if we have parameters
        gen_config = genai.GenerationConfig(**generation_config) if generation_config else None

        # Start chat with history (excluding the last message)
        chat = self.client.start_chat(
            history=gemini_messages[:-1] if len(gemini_messages) > 1 else []
        )

        # Send the last message with retry logic
        retry_delay = 60  # Initial retry delay
        while True:
            try:
                response = chat.send_message(
                    gemini_messages[-1]["parts"][0] if gemini_messages else "",
                    tools=gemini_tools,
                    generation_config=gen_config,
                )
                break  # Success, exit the loop
            except ResourceExhausted as e:
                print(f"Rate limit exceeded. Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
                retry_delay += 60  # Increase delay by 60 seconds for next attempt

        return response

    def _convert_schema_to_gemini(self, openai_schema):
        """Convert OpenAI JSON schema to Gemini schema format"""
        gemini_schema = genai.protos.Schema()

        if openai_schema.get("type") == "object":
            gemini_schema.type_ = genai.protos.Type.OBJECT

            if "properties" in openai_schema:
                for prop_name, prop_def in openai_schema["properties"].items():
                    prop_schema = genai.protos.Schema()

                    # Handle different property types
                    if prop_def.get("type") == "string":
                        prop_schema.type_ = genai.protos.Type.STRING
                    elif prop_def.get("type") == "integer":
                        prop_schema.type_ = genai.protos.Type.INTEGER
                    elif prop_def.get("type") == "number":
                        prop_schema.type_ = genai.protos.Type.NUMBER
                    elif prop_def.get("type") == "boolean":
                        prop_schema.type_ = genai.protos.Type.BOOLEAN
                    elif prop_def.get("type") == "array":
                        prop_schema.type_ = genai.protos.Type.ARRAY
                        if "items" in prop_def:
                            prop_schema.items = self._convert_schema_to_gemini(prop_def["items"])
                    elif prop_def.get("type") == "object":
                        prop_schema = self._convert_schema_to_gemini(prop_def)

                    if "description" in prop_def:
                        prop_schema.description = prop_def["description"]

                    if "enum" in prop_def:
                        prop_schema.enum = prop_def["enum"]

                    gemini_schema.properties[prop_name] = prop_schema

            if "required" in openai_schema:
                gemini_schema.required.extend(openai_schema["required"])

        return gemini_schema
