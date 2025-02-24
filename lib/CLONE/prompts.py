# --------------------------------------------------------------
# Stage 1: Emulating Experts through Role-Playing 
# -------------------------------------------------------------- 
ROLE_PLAYING_PROMPT = {
    "system": "You are a professional neuropsychologist who has made the given diagnosis. Answer the user's question.",
    "user": ("### Subject's Neuropsychological Test Results\n"
        "{snsb_score}\n\n"
        "### Your Rationale and Diagnosis\n"
        "{snsb_report}\n\n"
        "## Question\n"
        "How did you interpret the test results? What were the decisive factors for the diagnosis?"
    )
}

# --------------------------------------------------------------
# Stage 2: Synthesizing Step-by-step Diagnostic Guidelines 
# --------------------------------------------------------------
DRAFT_GUIDELINES_PROMPT = {
    "system": "You are a professional neuropsychologist developing step-by-step guidelines on how to interpret SNSB-C test results to determine the subject's cognitive level.",
    "user": ("Below is an explanation from a fellow neuropsychologist on why they interpreted the test results in that way and why they made such a diagnosis. "
        "Based on this, please create step-by-step guidelines for interpreting SNSB-C test results, focusing primarily on how to utilize objective statistical data. " 
        "Do not use your own knowledge or experience; only use the information provided.\n\n"
        "### Fellow Psychologist's Explanation\n"
        "{answer}"            
    )
}

UNIFIED_GUIDELINE_PROMPT = {
    "system": ("You are a professional neuropsychologist developing step-by-step guidelines on how to interpret SNSB-C test results to determine the subject's cognitive level falls among the following options:\n"
        "- (A) Normal cognition or Subjective cognitive decline\n"
        "- (B) Mild cognitive impairment, Early stage of dementia, or Alzheimer's disease"
    ),
    "user": ("Please consolidate the provided draft guidelines into a step-by-step interpretation of the SNSB-C test results. "
        "Do not use your own knowledge or experience, only the information provided. "
        "Organize the content in such a way that no examples from the draft guidelines are omitted."
        "Include interpretations for both options (A) and (B) as mentioned above."
        "Focus on using objective statistical data for the interpretation. "
        "Specifically, the method for making the final diagnosis should be described in as much detail as possible, including many different cases. "            
        "Now, please generate the step-by-step guidelines on how to interpret SNSB-C test results to determine the subject's cognitive level\n\n"
        "### Draft Guidelines\n"
        "{draft_guidelines}"
    )
}