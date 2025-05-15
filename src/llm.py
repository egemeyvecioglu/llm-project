from openai import OpenAI
from ollama import chat, ChatResponse, Client


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
        return OpenAI(**kwargs)

    def generate_response(self, messages: list, system_message: str = None, **kwargs):
        if system_message:
            messages.insert(0, {"role": "system", "content": system_message})

        response = self.client.chat.completions.create(
            model=self.model_name, messages=messages, **kwargs
        )

        return response


class OllamaClient(LLM):
    def _create_client(self, **kwargs):
        return Client(**kwargs)

    def generate_response(self, messages, system_message=None, **kwargs):

        if system_message:
            messages.insert(0, {"role": "system", "content": system_message})

        response: ChatResponse = self.client.chat(
            model=self.model_name, messages=messages, options=kwargs, stream=True
        )

        for chunk in response:
            print(chunk["message"]["content"], end="", flush=True)

        print(response["message"]["content"])
        # or access fields directly from the response object
        print(response.message.content)
