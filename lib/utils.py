import os
import random
import jsonlines

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def load_jsonl(file_path):
    """
    Load data from a JSON Lines file.
    """
    with jsonlines.open(file_path) as reader:
        return [record for record in reader]


def sample_source_set(info_dataset, source_ratio=0.1):
    """
    Splits the dataset into source and test sets based on the given source ratio.
    """
    # Split the dataset into two groups: HC (0) and MCI (1)   
    group_0 = [info_data["subject_id"] for info_data in info_dataset if info_data["group"] == 0]
    group_1 = [info_data["subject_id"] for info_data in info_dataset if info_data["group"] == 1]

    # Calculate the number of samples to be used as the source set
    source_size = round(len(info_dataset) * source_ratio)
    
    # Randomly sample subjects from each group
    random.seed(42)
    source_group_0 = random.sample(group_0, round(source_size // 2))
    source_group_1 = random.sample(group_1, round(source_size // 2))

    # Define the source set and the test set
    source_subject_ids = source_group_0 + source_group_1
    test_subject_ids = [info_data["subject_id"] for info_data in info_dataset]

    print(f"Number of source samples: {len(source_subject_ids)} (HC: {len(source_group_0)}, MCI: {len(source_group_1)})")
    print(f"Number of test samples: {len(test_subject_ids)} (Total subjects: {len(test_subject_ids)})")
    
    return source_subject_ids, test_subject_ids


def get_snsb_data_by_subject_id(subject_id, data_type):
    """
    Retrieve SNSB data for a given subject ID and data type.
    """
    dir_paths = {
        "score": "data/processed/SNSB/scores/",
        "report": "data/processed/SNSB/reports/eng/"
    }
    dir_path = dir_paths.get(data_type)
    
    for file_name in os.listdir(dir_path):
        if file_name.startswith(subject_id):
            with open(os.path.join(dir_path, file_name), 'r', encoding='utf-8') as file:
                return file.read()
    return None
    

def get_examples_for_fewshot(num_examples, source_subject_ids):
    """
    Get few-shot examples based on the number of examples requested.
    """
    # 1-shot
    if num_examples == 1:
        example_subject_ids = random.sample(source_subject_ids, num_examples)
        
        example_list = []
        for example_subject_id in example_subject_ids:
            snsb_scores = get_snsb_data_by_subject_id(example_subject_id, "score")
            snsb_report = get_snsb_data_by_subject_id(example_subject_id, "report")
            example_list.append(
                f"### SNSB-C Result\n{snsb_scores}\n\n"
                f"### Clinical Rationale\n{snsb_report.split('\n\n')[0]}\n\n"
                f"### Diagnosis\n{snsb_report.split('\n\n')[1]}"
            )
        
    # 2-shot or more
    elif num_examples > 1:
        info_dataset = load_jsonl("data/processed/info.jsonl")
        
        hc_subject_ids = []
        mci_subject_ids = []

        for source_subject_id in source_subject_ids:
            group = [info_data["group"] for info_data in info_dataset if info_data["subject_id"] == source_subject_id][0]
            if group == 0:
                hc_subject_ids.append(source_subject_id)
            elif group == 1:
                mci_subject_ids.append(source_subject_id)
        
        hc_sample_ids = random.sample(hc_subject_ids, num_examples // 2)
        hc_examples = []
        for hc_subject_id in hc_sample_ids:
            hc_snsb_scores = get_snsb_data_by_subject_id(hc_subject_id, "score")
            hc_snsb_report = get_snsb_data_by_subject_id(hc_subject_id, "report")
            hc_examples.append(
                f"### SNSB-C Result\n{hc_snsb_scores}\n\n"
                f"### Clinical Rationale\n{hc_snsb_report.split('\n\n')[0]}\n\n"
                f"### Diagnosis\n{hc_snsb_report.split('\n\n')[1]}"
            )

        mci_sample_ids = random.sample(mci_subject_ids, num_examples // 2)
        mci_examples = []
        for mci_subject_id in mci_sample_ids:
            mci_snsb_scores = get_snsb_data_by_subject_id(mci_subject_id, "score")
            mci_snsb_report = get_snsb_data_by_subject_id(mci_subject_id, "report")
            mci_examples.append(
                f"### SNSB-C Result\n{mci_snsb_scores}\n\n"
                f"### Clinical Rationale\n{mci_snsb_report.split('\n\n')[0]}\n\n"
                f"### Diagnosis\n{mci_snsb_report.split('\n\n')[1]}"
            )

        example_list = hc_examples + mci_examples
        random.shuffle(example_list)
    
    # Format the examples
    example_str = ""
    for i, example in enumerate(example_list):
        example_str += f"Example {i+1}.\n{example}\n\n"
        
    return example_str


def translate_text(text, source_language, target_language):
    """
    Translates text from the source language to the target language.
    """
    prompt = ChatPromptTemplate.from_template(
        "Translate the following {source_language} text to {target_language}: \n\n{text}\n\nTranslation:"
    )
    llm = ChatOpenAI(model="gpt-4o")
    translator = prompt | llm | StrOutputParser()
    
    return translator.invoke({
        "source_language": source_language,
        "target_language": target_language,
        "text": text
    })