import os
import json
import argparse
import pandas as pd
from tqdm import tqdm

from docling.document_converter import DocumentConverter

from lib.utils import load_jsonl, translate_text


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--info_data_path', type=str, default='data/raw/VEEM 대상자 정보.xlsx')
    parser.add_argument('--snsb_data_path', type=str, default='data/raw/SNSB/')
    parser.add_argument('--output_dir_path', type=str, default='data/processed')
    return parser.parse_args()


def preprocess_info_dataset(args, subject_id_list):
    """
    Preprocess the subjects' information dataset and save it to a JSONL file.
    """
    processed_info_file_path = os.path.join(args.output_dir_path, "info.jsonl")

    # Load the subjects' information dataset
    df_info = pd.read_excel(args.info_data_path, sheet_name='info')
    
    # Drop the duplicates and keep the recent record
    df_info = df_info.drop_duplicates(subset='subject_id', keep='last')
    
    # Write the processed information to a JSONL file
    num_report_none = 0
    with open(processed_info_file_path, 'w', encoding='utf-8') as f:
        for _, row in df_info.iterrows():
            subject_id = row['subject_id'].split('_')[1]
            
            # Skip the subject if the SNSB report is not found
            if (subject_id not in subject_id_list):
                print(f"Subject {subject_id}'s SNSB report not found")
                num_report_none += 1
                continue
            
            # Determine the group based on the diagnosis
            if pd.notna(row['진단결과']):
                if ('MCI' in row['진단결과']) or 'AD' in row['진단결과']:
                    group = 1
                else:
                    group = 0
            else:
                if ('MCI' in row['집단']) or 'AD' in row['집단']:
                    group = 1
                else:
                    group = 0
            
            # Save the processed information
            subject_info = {
                "subject_id" : subject_id,
                "name" : row['성명'],
                "age" : row['나이'],
                "gender" : 'M' if row['성별'] == '남성' else 'F',
                "education_years" : int(row['교육연한(y)']),
                "group": group
            }
            f.write(json.dumps(subject_info, ensure_ascii=False) + '\n')
    
    print(f"Total {num_report_none} subjects do not have SNSB report")
    return load_jsonl(processed_info_file_path)


def preprocess_snsb_score(args, info_dataset):
    """
    Preprocess SNSB score files by converting them from PDF to markdown format.
    """
    processed_snsb_score_dir_path = os.path.join(args.output_dir_path, "SNSB/scores")
    os.makedirs(processed_snsb_score_dir_path, exist_ok=True)
    
    # Initialize the document converter
    converter = DocumentConverter()
    
    # Define the directory paths
    snsb_score_dir_path = os.path.join(args.snsb_data_path, "scores")
    
    # Get the list of subject names from the info dataset
    subject_name_list = [info_data['name'] for info_data in info_dataset]
    for snsb_score_filename in tqdm(
        os.listdir(snsb_score_dir_path), desc="Processing SNSB scores",
    ):
        # Check if the file is a PDF file
        if not snsb_score_filename.endswith('.pdf'):
            continue
            
        # Get the subject name from the file name
        subject_name = snsb_score_filename.split('SNSB')[0].replace('_', '').strip()
        
        # Filter using the subject name
        if subject_name not in subject_name_list:
            print(f"Subject {subject_name} not found in the info dataset")
            continue
        
        # Get the subject ID from the info dataset
        subject_id = next(info_data['subject_id'] for info_data in info_dataset if info_data['name'] == subject_name)
        
        # ---------------------------------------------------------------------
        # Process the SNSB score
        # ---------------------------------------------------------------------
        snsb_score_file_path = os.path.join(snsb_score_dir_path, snsb_score_filename)
        
        # Define the file path for the processed SNSB score
        processed_snsb_score_file_path = os.path.join(
            processed_snsb_score_dir_path, f"{subject_id}.md"
        )
        
        # Check if the file already exists
        if os.path.exists(processed_snsb_score_file_path):
            print(f"File already exists: {processed_snsb_score_file_path}")
            continue
        print(f"Processing SNSB score for subject {subject_id}...")
        
        # Convert the document to markdown
        result = converter.convert(snsb_score_file_path)
        result_markdown = result.document.export_to_markdown()
        markdown_sections = result_markdown.split("\n\n")                
        
        # Save the extracted SNSB scores
        snsb_scores = "\n\n".join(markdown_sections[3:])        
        with open(processed_snsb_score_file_path, "w", encoding="utf-8") as md_file:
            md_file.write(snsb_scores)
                
    print(f"Total number of SNSB scores: {len(os.listdir(processed_snsb_score_dir_path))}")
    return None


def preprocess_snsb_report(args):
    """
    Preprocess SNSB reports by converting them to markdown format and translating them to English.
    """
    processed_snsb_report_dir_path = os.path.join(args.output_dir_path, "SNSB/reports")
    os.makedirs(f"{processed_snsb_report_dir_path}/kor", exist_ok=True)
    
    # Initialize the document converter
    converter = DocumentConverter()
    
    # Define the directory paths
    snsb_report_dir_path = os.path.join(args.snsb_data_path, "reports")
    
    # Get the list of subject IDs from the SNSB report directory
    subject_id_list = []
    for snsb_report_filename in tqdm(
        os.listdir(snsb_report_dir_path), desc="Processing SNSB reports",
    ):
        # Check if the file is not a docx file
        if not snsb_report_filename.endswith('.docx'):
            continue
            
        # Get the subject ID from the file name
        subject_id = snsb_report_filename.split('_')[2]
        subject_id_list.append(subject_id)
        
        # ----------------------------------------------------------------------
        # Process the SNSB report in Korean
        # ---------------------------------------------------------------------
        snsb_report_file_path = os.path.join(snsb_report_dir_path, snsb_report_filename)
        
        # Define the directory path for the processed SNSB report
        processed_snsb_report_kor_dir_path = f"{processed_snsb_report_dir_path}/kor"
        os.makedirs(processed_snsb_report_kor_dir_path, exist_ok=True)
        
        # Define the file path for the processed SNSB report
        processed_snsb_report_kor_file_path = os.path.join(
            processed_snsb_report_kor_dir_path, f"{subject_id}.md"
        )

        # Check if the file already exists
        if os.path.exists(processed_snsb_report_kor_file_path):
            print(f"File already exists: {processed_snsb_report_kor_file_path}")
            continue
        print(f"Processing SNSB report for subject {subject_id}...")
        
        # Convert the document to markdown
        result = converter.convert(snsb_report_file_path)
        result_markdown = result.document.export_to_markdown()
        markdown_sections = result_markdown.split("\n\n")                
        
        # Extract the rationales and diagnosis
        start_idx = markdown_sections.index("Neuropsychological Assessment Summary & Conclusions") + 1
        end_idx = next(i for i, section in enumerate(markdown_sections) if "| K-MMSE-2" in section)
        
        # Save the extracted rationales and diagnosis
        diagnosis_with_rationales_kor = "\n\n".join(markdown_sections[start_idx:end_idx])
        with open(processed_snsb_report_kor_file_path, "w", encoding="utf-8") as md_file:
            md_file.write(diagnosis_with_rationales_kor)
        
        # ----------------------------------------------------------------------
        # Process the SNSB report in English
        # ---------------------------------------------------------------------
        processed_snsb_report_eng_dir_path = os.path.join(processed_snsb_report_dir_path, "eng")
        os.makedirs(processed_snsb_report_eng_dir_path, exist_ok=True)
        
        # Define the file path for the processed SNSB report
        processed_snsb_report_eng_file_path = os.path.join(
            processed_snsb_report_eng_dir_path, f"{subject_id}.md"
        )
        
        # Check if the file already exists
        if os.path.exists(processed_snsb_report_eng_file_path):
            print(f"File already exists: {processed_snsb_report_eng_file_path}")
            continue
        
        # Translate the Korean text to English
        diagnosis_with_rationales_eng = translate_text(
            text=diagnosis_with_rationales_kor,
            source_language="Korean",
            target_language="English"
        )
        
        # Save the translated text
        with open(processed_snsb_report_eng_file_path, "w", encoding="utf-8") as md_file:
            md_file.write(diagnosis_with_rationales_eng)
    
    print(f"Total number of SNSB reports: {len(os.listdir(processed_snsb_report_kor_dir_path))}")
    return subject_id_list


def validate_data_consistency(args, subject_id_list, info_dataset):
    subject_ids = sorted(subject_id_list)
    info_subject_ids = sorted([info['subject_id'] for info in info_dataset])
    report_subject_ids = sorted([f.split('.')[0] for f in os.listdir(f"{args.output_dir_path}/SNSB/reports/eng")])
    score_subject_ids = sorted([f.split('.')[0] for f in os.listdir(f"{args.output_dir_path}/SNSB/scores")])
    assert subject_ids == info_subject_ids == report_subject_ids == score_subject_ids
    

def main(args):
    os.makedirs(args.output_dir_path, exist_ok=True)
    
    # Preprocess the SNSB reports and scores
    subject_id_list = preprocess_snsb_report(args)
    info_dataset = preprocess_info_dataset(args, subject_id_list)
    _ = preprocess_snsb_score(args, info_dataset)
    
    # Check if the subject IDs are consistent across the datasets
    validate_data_consistency(args, subject_id_list, info_dataset)


if __name__ == '__main__':
    args = get_args()
    main(args)
