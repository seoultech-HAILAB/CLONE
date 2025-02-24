import os
import jsonlines
from tqdm import tqdm

from lib.CLONE.workflow import DiagnosticGuidelineSynthesizer
from lib.utils import get_snsb_data_by_subject_id, get_examples_for_fewshot


class AbstractPayloadCreator:
    """
    An abstract class for payload creators.
    """
    def __init__(self, temperature, system_prompt_path, user_prompt_template_path):
        self.temperature = temperature
        self.system_prompt = self._load_prompt(system_prompt_path)
        self.prompt_template = self._load_prompt(user_prompt_template_path)
    
    def create_payload(self, **kwargs):
        """
        Abstract method to create a payload for an API request.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def save_payload(self, payload, payload_path):
        """
        Save the given payload to the specified file path.
        """
        os.makedirs(os.path.dirname(payload_path), exist_ok=True)
        with jsonlines.open(payload_path, mode="a") as writer:
            writer.write(payload)
            
    def load_cached_payload(self, payloads_path):
        """
        Load the cached payloads from the specified file path.
        """
        print("Checking for cached payloads...")
        if os.path.exists(payloads_path):
            print(f"Payloads already exist at '{payloads_path}'.")
            with jsonlines.open(payloads_path, mode="r") as reader:
                return list(reader)
        print("No cached payloads found.")
        return []
    
    @staticmethod
    def _load_prompt(prompt_path):
        """
        Load the prompt text from the specified file path.
        """
        if prompt_path and os.path.isfile(prompt_path):
            with open(prompt_path, "r") as f:
                return f.read()
        return ""
    
    
class PayloadCreator(AbstractPayloadCreator):
    def __init__(self, temperature, system_prompt_path, user_prompt_template_path):
        super().__init__(temperature, system_prompt_path, user_prompt_template_path)

    def process_payloads(self, info_dataset, payload_path, test_subject_ids, context_generator):
        """
        Processes payloads by either loading cached payloads or creating new ones.
        Args:
            info_dataset (list): List of information data for each subject.
            payload_path (str): Path to the cached payloads.
            test_subject_ids (set): Set of subject IDs to be marked as test subjects.
            context_generator (function): Function to generate context for each info data.
        Returns:
            list: List of processed payloads.
        """
        # ---------------------------------------------------------------------
        # Check to cached payloads
        # ---------------------------------------------------------------------
        num_subjects = len(info_dataset)
        payload_list = self.load_cached_payload(payload_path)
        num_cached = len(payload_list)

        print(f"- Num of subjects: {num_subjects}")
        print(f"- Num of payloads: {num_cached}")

        if num_cached == num_subjects:
            print("Successfully loaded the cached payloads!")
            return payload_list
        elif num_cached < num_subjects:
            print(f"Continuing from {num_cached} cached payloads...")

        # ---------------------------------------------------------------------
        # Create the payloads
        # ---------------------------------------------------------------------
        for info_data in tqdm(info_dataset[num_cached:], desc="Creating payloads"):
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.prompt_template.format(context=context_generator(info_data))},
            ]
            payload = {
                "subject_id": info_data["subject_id"],
                "ground_truth": info_data["group"],
                "test": 1 if info_data["subject_id"] in test_subject_ids else 0,
                "messages": messages,
                "temperature": self.temperature,
            }
            self.save_payload(payload, payload_path)
            payload_list.append(payload)
        return payload_list


class ZeroShotPayloadCreator(PayloadCreator):
    """
    A class to create zero-shot payloads for the API request.
    """
    def create_payload(self, **kwargs):
        return self.process_payloads(
            kwargs['info_dataset'], kwargs['payload_path'], kwargs['test_subject_ids'],
            lambda info_data: (
                f"### Subject information\n"
                f"Age: {info_data['age']}\n"
                f"Gender: {info_data['gender']}\n"
                f"Education years: {info_data['education_years']}\n\n"
                "### SNSB-C Result\n"
                + get_snsb_data_by_subject_id(info_data['subject_id'], data_type="score")
            )
        )
            
    
class FewShotPayloadCreator(PayloadCreator):
    """
    A class to create few-shot payloads for the API request.
    """
    def create_payload(self, **kwargs):
        return self.process_payloads(
            kwargs['info_dataset'], kwargs['payload_path'], kwargs['test_subject_ids'],
            lambda info_data: (
                "### Examples\n"
                + get_examples_for_fewshot(kwargs['num_examples'], kwargs['source_subject_ids']) + "\n\n"
                f"### Subject information\n"
                f"Age: {info_data['age']}\n"
                f"Gender: {info_data['gender']}\n"
                f"Education years: {info_data['education_years']}\n\n"
                "### SNSB-C Result\n"
                + get_snsb_data_by_subject_id(info_data['subject_id'], data_type="score")
            )
        )

class CustomPayloadCreator(PayloadCreator):
    """
    A class to create custom payloads for the API request.
    """
    def create_payload(self, **kwargs):
        # CLONE: Synthesize diagnostic guideline
        diagnostic_guideline = DiagnosticGuidelineSynthesizer(
            kwargs['model']
        ).synthesize_diagonstic_guideline(
            kwargs['source_subject_ids']
        )
        return self.process_payloads(
            kwargs['info_dataset'], kwargs['payload_path'], kwargs['test_subject_ids'],
            lambda info_data: (
                diagnostic_guideline + "\n\n"
                f"### Subject information\n"
                f"Age: {info_data['age']}\n"
                f"Gender: {info_data['gender']}\n"
                f"Education years: {info_data['education_years']}\n\n"
                "### SNSB-C Result\n"
                + get_snsb_data_by_subject_id(info_data['subject_id'], data_type="score")
            )
        )
    

class PayloadCreatorFactory:
    """
    A factory class to specify payload creator based on the test type.
    """
    @staticmethod
    def get_payload_creator(test_type, temperature, system_prompt_path, user_prompt_template_path):
        if test_type == "zeroshot":
            return ZeroShotPayloadCreator(temperature, system_prompt_path, user_prompt_template_path)
        elif test_type == "fewshot":
            return FewShotPayloadCreator(temperature, system_prompt_path, user_prompt_template_path)
        elif test_type == "custom":
            return CustomPayloadCreator(temperature, system_prompt_path, user_prompt_template_path)
        else:
            raise ValueError(f"Unsupported task type: {test_type}")