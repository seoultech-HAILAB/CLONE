import os
import json
import jsonlines
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from prometheus_eval import PrometheusEval
from prometheus_eval.vllm import VLLM
from prometheus_eval.prompts import ABSOLUTE_PROMPT

from lib.utils import load_jsonl, get_snsb_data_by_subject_id
from lib.rubrics import RATIONALE_RUBRICS


class AbstractResponseEvaluator:
    """
    An abstract class for evaluation handlers.
    """
    def __init__(self):
        self.num_results = 0
    
    def evaluate_response(self, **kwargs):
        """
        Evaluate the submission and return the score.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def save_results(self, eval_result, results_path):
        """
        Save the evaluation results to a file.
        """
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        if isinstance(eval_result, dict):
            with jsonlines.open(results_path, mode="a") as writer:
                writer.write(eval_result)
        elif isinstance(eval_result, list):
            with jsonlines.open(results_path, mode="a") as writer:
                writer.write_all(eval_result)
        else:
            raise ValueError("Unsupported evaluation result format.")
            
    def load_cached_results(self, results_path):
        """
        Load the evaluation results from a file.
        """
        print("Checking for cached evaluation results...")
        if os.path.exists(results_path):
            print(f"Evaluation results already exist at '{results_path}'.")
            with open(results_path, "r") as f:
                return json.load(f)
        print("No cached evaluation results found.")
        return []


class ResponseEvaluator(AbstractResponseEvaluator):
    def __init__(self):
        super().__init__()

    def process_evaluation(self, input_payloads, response_list, results_path, eval_method):
        """
        Processes the evaluation of responses.
        Args:
            input_payloads (list): The input payloads to be evaluated.
            response_list (list): The list of responses to be evaluated.
            results_path (str): The path to save or load cached evaluation results.
            eval_method (callable): The method used to evaluate the responses.
        Returns:
            dict: The evaluation results.
        """
        # ---------------------------------------------------------------------
        # Check to cached evaluation results
        # ---------------------------------------------------------------------
        eval_result = self.load_cached_results(results_path)
        if eval_result:
            print("Successfully loaded the cached evaluation results!")
            return eval_result
        
        # ---------------------------------------------------------------------
        # Evaluate the responses
        # ---------------------------------------------------------------------
        eval_result = eval_method(input_payloads, response_list)
        self.save_results(eval_result, results_path)
        return eval_result
    

class ClassificationResponseEvaluator(ResponseEvaluator):
    """
    A class to evaluate the classification responses.
    """
    def evaluate_response(self, **kwargs):
        return self.process_evaluation(
            kwargs['input_payloads'], kwargs['response_list'], kwargs['results_path'],
            lambda inputs, responses: self._evaluate_classification(inputs, responses, kwargs['test_subject_ids'])
        )

    def _evaluate_classification(self, input_payloads, response_list, test_subject_ids):
        y_true, y_pred = [], []
        for test_subject_id in tqdm(test_subject_ids, desc="Evaluating responses"):
            ground_truth = next(p["ground_truth"] for p in input_payloads if p["subject_id"] == test_subject_id)
            diagnosis = next(1 if "(B)" in r["diagnosis"] else 0 for r in response_list if r["subject_id"] == test_subject_id)
            y_true.append(ground_truth)
            y_pred.append(diagnosis)
        
        # Calculate the evaluation metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics = {
            "accuracy": (tp + tn) / (tp + tn + fp + fn),
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "f1_score": (2 * tp / (2 * tp + fp + fn)) if (tp + fp + fn) > 0 else 0
        }
        return {"total_samples": len(y_true), **{k: round(v, 4) for k, v in metrics.items()}}


class RubricResponseEvaluator(ResponseEvaluator):
    """
    A class to evaluate the responses using rubrics.
    """
    def evaluate_response(self, **kwargs):
        return self.process_evaluation(
            kwargs['input_payloads'], kwargs['response_list'], kwargs['results_path'],
            lambda inputs, responses: self._evaluate_rubric(inputs, responses, kwargs['baseline_response_path'])
        )

    def _evaluate_rubric(self, input_payloads, response_list, baseline_response_path):
        baseline_list = load_jsonl(baseline_response_path)
        
        # Initialize the Prometheus as the evaluator
        model = VLLM(model="prometheus-eval/prometheus-8x7b-v2.0", tensor_parallel_size=4, enforce_eager=True)
        judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)
        
        # Prepare the batch instructions, responses, and references
        batch_instructions, batch_responses, batch_references = [], [], []
        for input, output in zip(input_payloads, response_list):
            batch_instructions.append(input["messages"][0]["content"] + "\n\n" + get_snsb_data_by_subject_id(input["subject_id"], "score"))
            batch_responses.append(output["generated_response"])
            batch_references.append(get_snsb_data_by_subject_id(input["subject_id"], "report"))
        
        # Evaluate the responses using rubrics
        rubric_results = []
        for criteria, rubric in RATIONALE_RUBRICS.items():
            feedbacks, scores = judge.relative_grade(
                instructions=batch_instructions,
                responses_A=batch_responses,
                responses_B=baseline_list,
                rubric=rubric,
                reference_answers=batch_references
            )
            rubric_results.extend([{"criteria": criteria, "feedback": fb, "score": sc} for fb, sc in zip(feedbacks, scores)])
        return rubric_results
                

class ResponseEvaluatorFactory:
    """
    A factory class to specify evaluator based on the type of evaluation.
    """
    @staticmethod
    def get_evaluator(eval_type):
        if eval_type == "clf":
            return ClassificationResponseEvaluator()
        elif eval_type == "rubric":
            return RubricResponseEvaluator()
        else:
            raise ValueError(f"Unsupported evaluation type: {eval_type}.")