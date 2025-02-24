import os
import jsonlines
from tqdm import tqdm
from openai import OpenAI
from ollama import chat, ChatResponse


class AbstractAPIExecutor:
    """
    An abstract class for API executors.
    """
    def __init__(self, model, api_key):
        self.model = model
        self.api_key = api_key
    
    def fetch_response(self, **kwargs):
        """
        Abstract method to fetch the response from the API.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def save_response(self, response, response_path):
        """
        Save the response to a JSON Lines file.
        """
        os.makedirs(os.path.dirname(response_path), exist_ok=True)
        with jsonlines.open(response_path, mode="a") as writer:
            writer.write(response)
    
    def load_cached_response(self, response_path):
        """
        Load the cached responses from the specified file path.
        """
        print("Checking for cached responses...")
        if os.path.exists(response_path):
            print(f"Responses already exist at '{response_path}'.")
            with jsonlines.open(response_path, mode="r") as reader:
                return list(reader)
        print("No cached responses found.")
        return []


class APIExecutor(AbstractAPIExecutor):
    def __init__(self, model, api_key):
        super().__init__(model, api_key)

    def process_responses(self, input_payloads, test_subject_ids, response_path, fetch_method):
        """
        Processes responses by either loading cached responses or fetching new ones.
        Args:
            input_payloads (list): List of payloads to be sent to the API.
            test_subject_ids (list): List of test subject IDs.
            response_path (str): Path to the cached responses.
            fetch_method (callable): Method to fetch responses from the API.
        Returns:
            list: List of responses.
        """
        # ---------------------------------------------------------------------
        # Check to cached response
        # ---------------------------------------------------------------------
        response_list = self.load_cached_response(response_path)
        num_responses = len(response_list)

        print(f"- Num of testset: {len(test_subject_ids)}")
        print(f"- Num of responses: {num_responses}")

        if num_responses == len(test_subject_ids):
            print("Successfully loaded the cached responses!")
            return response_list
        elif num_responses > 0:
            print(f"Continuing from {num_responses} cached responses...")

        # ---------------------------------------------------------------------
        # Execute the API
        # ---------------------------------------------------------------------
        for test_subject_id in tqdm(
            test_subject_ids[num_responses:], desc="Fetching responses"
        ):
            payload = next(p for p in input_payloads if p["subject_id"] == test_subject_id)
            response = fetch_method(payload)
            response_list.append(response)
            self.save_response(response, response_path)
        return response_list
    

class OpenaiAPIExecutor(APIExecutor):
    """
    A class to execute the OpenAI API.
    """
    def __init__(self, model, api_key):
        super().__init__(model, api_key, 0)
        self.client = OpenAI(api_key=api_key)
    
    def fetch_response(self, **kwargs):
        return self.process_responses(
            kwargs['input_payloads'], kwargs['test_subject_ids'], kwargs['response_path'],
            lambda payload: self._fetch_openai_response(payload)
        )
        
    def _fetch_openai_response(self, payload):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=payload["messages"],
                temperature=payload["temperature"]
            )
            response = completion.choices[0].message.content
            return {
                "subject_id": payload["subject_id"],
                "diagnosis": response.split("### Diagnosis")[1].strip(),
                "generated_response": response,
            }
        except Exception as e:
            print(f"Error during fetching response: {e}")
            return {}
    

class OllamaAPIExecutor(APIExecutor):
    """
    A class to execute the Ollama API.
    """
    def __init__(self, model, api_key):
        super().__init__(model, api_key)
        # self.client = OpenAI(
        #     base_url="http://localhost:11434/v1",
        #     api_key=self.api_key,
        # )
        
    def fetch_response(self, **kwargs):
        return self.process_responses(
            kwargs['input_payloads'], kwargs['test_subject_ids'], kwargs['response_path'],
            lambda payload: self._fetch_ollama_response(payload)
        )

    def _fetch_ollama_response(self, payload):
        try:
            completion: ChatResponse = chat(
                model=self.model,
                messages=payload["messages"],
                options={"temperature": payload["temperature"], "num_ctx": 8192}
            )
            response = completion.message.content
            return {
                "subject_id": payload["subject_id"],
                "diagnosis": response.split("### Diagnosis")[1].strip(),
                "generated_response": response,
            }
        except Exception as e:
            print(f"Error during fetching response: {e}")
            return {}
    

class VllmAPIExecutor(APIExecutor):
    """
    A class to execute the vLLM API.
    """
    def __init__(self, model, api_key):
        super().__init__(model, api_key)
        self.client = OpenAI(
            base_url="http://localhost:8000/v1",
        )

    
class APIExecutorFactory:
    """
    A factory class to specify API executor based on the API type.
    """
    @staticmethod
    def get_api_executor(model, api_type, api_key):
        if api_type == 'openai':
            return OpenaiAPIExecutor(model, api_key)
        elif api_type == 'ollama':
            return OllamaAPIExecutor(model, api_key)
        elif api_type == 'vllm':
            return VllmAPIExecutor(model, api_key)
        else:
            raise ValueError(f"Unsupported API type: {api_type}.")
        
        