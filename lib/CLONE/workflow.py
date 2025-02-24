import os

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .prompts import (
    ROLE_PLAYING_PROMPT, 
    DRAFT_GUIDELINES_PROMPT, 
    UNIFIED_GUIDELINE_PROMPT
)
from lib.utils import get_snsb_data_by_subject_id


class DiagnosticGuidelineSynthesizer:
    def __init__(self, model: str = "llama3.3:70b", num_ctx: int = 8192):
        self.model = model
        self.llm = self._initialize_llm(model, num_ctx)
        self.save_path = f"results/guidelines/{model}.txt"

        # Chains for each stage of the guideline synthesis process
        self.roleplaying_chain = self._get_chain(ROLE_PLAYING_PROMPT)
        self.drafting_chain = self._get_chain(DRAFT_GUIDELINES_PROMPT)
        self.writing_chain = self._get_chain(UNIFIED_GUIDELINE_PROMPT)
    
    def synthesize_diagonstic_guideline(self, source_subject_ids):
        """
        Synthesizes diagnostic guidelines based on SNSB-C test results for given subject IDs.
        Args:
            source_subject_ids (list): List of subject IDs to process.
        Returns:
            str: Consolidated diagnostic guideline.
        """
        if self._guideline_exists():
            return self._load_existing_guideline()
        
        print("Synthesizing diagnostic guidelines...")
        snsb_scores, snsb_reports = self._load_snsb_data(source_subject_ids)      
        
        # ---------------------------------------------------------------------
        # Stage 1: Emulating Experts through Role-Playing
        # ---------------------------------------------------------------------
        answer_set = self.roleplaying_chain.batch(
            [
                {"snsb_score": snsb_score, "snsb_report": snsb_report}
                for snsb_score, snsb_report in zip(snsb_scores, snsb_reports)
            ]
        )
        
        # ---------------------------------------------------------------------
        # Stage 2: Synthesizing Step-by-Step Diagnostic Guidelines
        # ---------------------------------------------------------------------
        drafts = self.drafting_chain.batch(
            [
                {"answer": answer} for answer in answer_set
            ]
        )
        
        # Consolidate the draft guidelines into a unified guideline
        diagnostic_guideline = self.writing_chain.invoke(
            {"draft_guidelines": "\n\n".join(drafts)}
        )
        
        # Save the generated guideline
        self._save_guideline(diagnostic_guideline)
        
        return diagnostic_guideline


    def _initialize_llm(self, model, num_ctx):
        """
        Initializes the appropriate language model based on the model.
        """
        if model.startswith("gpt"):
            return ChatOpenAI(model=model, temperature=0.1)
        return ChatOllama(model=model, temperature=0.1, num_ctx=num_ctx)
    
    def _get_chain(self, prompt_text):
        """
        Creates a chain using the given prompt text.
        """
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_text['system']),
                ("user", prompt_text['user'])
            ]
        )
        return prompt_template | self.llm | StrOutputParser()
    
    def _guideline_exists(self):
        """
        Checks if the guideline already exists.
        """
        return os.path.exists(self.save_path)
    
    def _load_existing_guideline(self):
        """
        Loads the existing guideline from the file.
        """
        print("Diagnostic guidelines have already been generated.")
        with open(self.save_path, "r") as f:
            return f.read()
    
    def _load_snsb_data(self, subject_ids):
        """
        Loads SNSB scores and reports for the given subject IDs.
        """
        scores, reports = [], []
        for subject_id in subject_ids:
            scores.append(get_snsb_data_by_subject_id(subject_id, "score"))
            reports.append(get_snsb_data_by_subject_id(subject_id, "report"))
        return scores, reports
    
    def _save_guideline(self, guideline):
        """
        Saves the synthesized diagnostic guideline to a file.
        """
        with open(self.save_path, "w") as f:
            f.write(guideline)