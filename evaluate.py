import os
import argparse

from lib.payload_creator import PayloadCreatorFactory
from lib.api_executor import APIExecutorFactory
from lib.response_evaluator import ResponseEvaluatorFactory
from lib.utils import load_jsonl, sample_source_set


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_type", type=str, default="zeroshot", choices=['zeroshot', 'fewshot', 'custom'])
    parser.add_argument("--num_examples", type=int, default=0)
    parser.add_argument("--source_ratio", type=float, default=0.1)
    parser.add_argument("--api_type", type=str, default="ollama", choices=['openai', 'ollama', 'vllm'])
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--model", type=str, default="llama3.3:70b")
    parser.add_argument("--batch", type=bool, default=False)
    parser.add_argument("--eval_type", type=str, default="clf", choices=['clf', 'rubric'])
    parser.add_argument("--baseline_type", type=str, default="zeroshot", choices=['zeroshot', 'fewshot1', 'fewshot2'])
    return parser.parse_args()


def main(args):
    # ----------------------------------------------------------------------
    # Define the paths
    # ----------------------------------------------------------------------
    REPO_PATH = os.path.abspath(os.getcwd())
    
    SYSTEM_PROMPT_PATH = f"{REPO_PATH}/prompts/system.txt"
    USER_PROMPT_TEMPLATE_PATH = f"{REPO_PATH}/prompts/user.txt"    
    INFO_DATASET_PATH = f"{REPO_PATH}/data/processed/info.jsonl"
    
    if args.test_type == "fewshot":
        TEST_PREFIX = f"{args.test_type}{args.num_examples}-{args.model}"
    else:
        TEST_PREFIX = f"{args.test_type}-{args.model}"
    
    if args.test_type == "custom":
        input_payload_path = f'{REPO_PATH}/data/payloads/{TEST_PREFIX}.jsonl'
    else:
        input_payload_path = f'{REPO_PATH}/data/payloads/{TEST_PREFIX.split('-')[0]}.jsonl'
        
    if args.eval_type == "rubric":
        baseline_response_path = f'{REPO_PATH}/results/{args.baseline_type}-{args.model}.output.jsonl'
    else:
        baseline_response_path = None
        
    output_path = f'{REPO_PATH}/results/{TEST_PREFIX}.output.jsonl'
    eval_results_path = f'{REPO_PATH}/results/{TEST_PREFIX}.eval_results-{args.eval_type}.jsonl'
    
    # ----------------------------------------------------------------------
    # Load the dataset
    # ----------------------------------------------------------------------
    info_dataset = load_jsonl(INFO_DATASET_PATH)
    
    source_subject_ids, test_subject_ids = sample_source_set(info_dataset, source_ratio=args.source_ratio)
    
    # ----------------------------------------------------------------------
    # Create the payloads
    # ----------------------------------------------------------------------
    input_payloads = PayloadCreatorFactory.get_payload_creator(
        test_type=args.test_type,
        temperature=args.temperature,
        system_prompt_path=SYSTEM_PROMPT_PATH,
        user_prompt_template_path=USER_PROMPT_TEMPLATE_PATH
    ).create_payload(
        info_dataset=info_dataset,
        payload_path=input_payload_path,
        num_examples=args.num_examples,
        model=args.model,
        source_subject_ids=source_subject_ids,
        test_subject_ids=test_subject_ids,
    )
    
    # ----------------------------------------------------------------------
    # Execute the API
    # ----------------------------------------------------------------------
    response_list = APIExecutorFactory.get_api_executor(
        model=args.model,
        api_type=args.api_type,
        api_key=args.api_key,
    ).fetch_response(
        input_payloads=input_payloads,
        test_subject_ids=test_subject_ids,
        response_path=output_path,
        batch=args.batch,
    )
    
    # ----------------------------------------------------------------------
    # Evaluate the responses
    # ----------------------------------------------------------------------
    eval_results = ResponseEvaluatorFactory.get_evaluator(
        eval_type=args.eval_type
    ).evaluate_response(
        input_payloads=input_payloads,
        response_list=response_list,
        test_subject_ids=test_subject_ids,
        baseline_response_path=baseline_response_path,
        results_path=eval_results_path,
    )
    
    # Check the evaluation results
    print(eval_results)
    

if __name__ == "__main__":
    args = get_args()
    main(args)