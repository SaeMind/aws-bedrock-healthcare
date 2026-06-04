"""
prompt_templates.py
Clinical prompt templates for AWS Bedrock inference.
Author: Andrew Lee | UTHealth Houston SBMI
"""

from typing import Literal

TaskType = Literal["discharge_summary", "risk_narrative", "icd_classification", "care_gap_analysis"]


def build_clinical_prompt(patient_data: dict, task: TaskType) -> str:
    """Build task-specific clinical prompt from structured patient data."""
    if task == "discharge_summary":
        return _discharge_summary_prompt(patient_data)
    elif task == "risk_narrative":
        return _risk_narrative_prompt(patient_data)
    elif task == "icd_classification":
        return _icd_classification_prompt(patient_data)
    elif task == "care_gap_analysis":
        return _care_gap_analysis_prompt(patient_data)
    else:
        raise ValueError(f"Unknown task type: {task}")


def _discharge_summary_prompt(d: dict) -> str:
    return f"""Generate a 2-3 sentence patient-friendly discharge summary based on the following structured clinical data only. Use plain language (8th grade reading level). Do not add clinical information not explicitly stated in the data.

STRUCTURED CLINICAL DATA:
- Patient age: {d.get('age', 'unknown')}
- Admission date: {d.get('admission_date', 'unknown')}
- Discharge date: {d.get('discharge_date', 'unknown')}
- Length of stay: {d.get('los_days', 'unknown')} days
- Principal diagnosis (ICD-10): {d.get('principal_dx', 'unknown')}
- MS-DRG: {d.get('drg_code', 'unknown')}
- Discharge status: {d.get('discharge_status', 'home')}

PATIENT-FRIENDLY DISCHARGE SUMMARY:"""


def _risk_narrative_prompt(d: dict) -> str:
    return f"""Based on the following patient risk assessment data, write a 2-sentence clinical risk summary for care coordination. Be factual. Do not recommend specific medications or treatments.

RISK DATA:
- Readmission risk score: {d.get('risk_score', 'unknown')} (scale 0-1)
- Risk tier: {d.get('risk_tier', 'unknown')} (low/medium/high)
- Primary diagnosis: {d.get('principal_dx', 'unknown')}
- Length of stay: {d.get('los_days', 'unknown')} days
- Age: {d.get('age', 'unknown')}

CLINICAL RISK SUMMARY:"""


def _icd_classification_prompt(d: dict) -> str:
    codes = ", ".join(d.get("icd_codes", []))
    return f"""Classify the following ICD-10-CM diagnosis codes into the most appropriate clinical domain. Choose one domain from: [cardiovascular, respiratory, endocrine/metabolic, infectious, neurological, oncology, musculoskeletal, renal, gastrointestinal, psychiatric, trauma, other].

ICD-10-CM codes: {codes}

Respond with JSON only: {{"primary_domain": "<domain>", "secondary_domain": "<domain or null>", "confidence": <0.0-1.0>}}

JSON:"""


def _care_gap_analysis_prompt(d: dict) -> str:
    return f"""Based on the following patient clinical profile, identify 2-3 potential care gaps or follow-up actions that a care coordinator should review. Be specific and actionable. Base your response only on the provided data.

PATIENT PROFILE:
- Age: {d.get('age', 'unknown')}
- Principal diagnosis: {d.get('principal_dx', 'unknown')}
- Recent hospitalization: {d.get('los_days', 'unknown')} days
- Risk tier: {d.get('risk_tier', 'unknown')}
- Active conditions: {', '.join(d.get('conditions', ['unknown']))}
- Current medications: {', '.join(d.get('medications', ['unknown']))}

CARE GAPS AND FOLLOW-UP ACTIONS:"""
