"""
lambda_handler.py
AWS Lambda function — Bedrock Healthcare API Gateway handler.
Author: Andrew Lee | UTHealth Houston SBMI

Routes:
  POST /narrative       Generate clinical narrative for patient record
  POST /rag/query       RAG-powered clinical knowledge base query
  POST /classify        ICD-10 code classification
  GET  /health          Health check
"""

import json
import logging
import os
from typing import Any

from src.bedrock.bedrock_client import BedrockClient
from src.bedrock.prompt_templates import build_clinical_prompt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

KNOWLEDGE_BASE_ID = os.environ.get("BEDROCK_KB_ID", "")
client = BedrockClient(region=os.environ.get("AWS_REGION", "us-east-1"))


def lambda_handler(event: dict, context: Any) -> dict:
    """Main Lambda handler — routes to appropriate function."""
    path   = event.get("path", "/")
    method = event.get("httpMethod", "GET")
    body   = {}

    if event.get("body"):
        try:
            body = json.loads(event["body"])
        except json.JSONDecodeError:
            return _error_response(400, "Invalid JSON body")

    logger.info(f"Request: {method} {path}")

    try:
        if path == "/health" and method == "GET":
            return _success_response({"status": "healthy", "service": "bedrock-healthcare-api"})

        elif path == "/narrative" and method == "POST":
            return _handle_narrative(body)

        elif path == "/rag/query" and method == "POST":
            return _handle_rag_query(body)

        elif path == "/classify" and method == "POST":
            return _handle_classify(body)

        else:
            return _error_response(404, f"Route not found: {method} {path}")

    except Exception as e:
        logger.exception(f"Unhandled error: {e}")
        return _error_response(500, "Internal server error")


def _handle_narrative(body: dict) -> dict:
    """Generate clinical narrative from structured patient data."""
    patient_data = body.get("patient_data")
    task         = body.get("task", "discharge_summary")

    if not patient_data:
        return _error_response(400, "Missing required field: patient_data")
    if task not in ("discharge_summary", "risk_narrative", "care_gap_analysis"):
        return _error_response(400, f"Invalid task: {task}")

    narrative = client.generate_clinical_narrative(patient_data, task)
    return _success_response({
        "narrative": narrative,
        "task":      task,
        "model":     "claude-3-sonnet",
    })


def _handle_rag_query(body: dict) -> dict:
    """Answer clinical question using Bedrock RAG knowledge base."""
    query = body.get("query")
    if not query:
        return _error_response(400, "Missing required field: query")
    if not KNOWLEDGE_BASE_ID:
        return _error_response(503, "Knowledge base not configured")

    result = client.retrieve_and_generate(
        query              = query,
        knowledge_base_id  = KNOWLEDGE_BASE_ID,
        max_results        = body.get("max_results", 5),
    )
    return _success_response({
        "answer":    result["answer"],
        "citations": len(result.get("citations", [])),
        "session_id": result.get("session_id"),
    })


def _handle_classify(body: dict) -> dict:
    """Classify ICD-10 codes into clinical domain using Claude."""
    icd_codes = body.get("icd_codes", [])
    if not icd_codes:
        return _error_response(400, "Missing required field: icd_codes")

    prompt = build_clinical_prompt({"icd_codes": icd_codes}, "icd_classification")
    raw    = client.invoke_claude(prompt=prompt, max_tokens=200, temperature=0.0)

    try:
        classification = json.loads(raw.strip())
    except json.JSONDecodeError:
        classification = {"raw_response": raw}

    return _success_response({"classification": classification, "input_codes": icd_codes})


def _success_response(body: dict, status_code: int = 200) -> dict:
    return {
        "statusCode": status_code,
        "headers":    {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
        "body":       json.dumps(body),
    }


def _error_response(status_code: int, message: str) -> dict:
    return {
        "statusCode": status_code,
        "headers":    {"Content-Type": "application/json"},
        "body":       json.dumps({"error": message}),
    }
