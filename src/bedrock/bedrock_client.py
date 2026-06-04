"""
bedrock_client.py
AWS Bedrock API wrapper for clinical AI inference.
Author: Andrew Lee | UTHealth Houston SBMI

Supports:
  - InvokeModel (synchronous, single-turn)
  - InvokeModelWithResponseStream (streaming)
  - Agents Runtime (multi-turn RAG with KB retrieval)
"""

import json
import logging
import os
from typing import Generator, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

REGION = os.environ.get("AWS_REGION", "us-east-1")

# Model IDs — Bedrock ARN format
CLAUDE_3_SONNET  = "anthropic.claude-3-sonnet-20240229-v1:0"
CLAUDE_3_HAIKU   = "anthropic.claude-3-haiku-20240307-v1:0"
TITAN_EMBED_V2   = "amazon.titan-embed-text-v2:0"
TITAN_TEXT_LITE  = "amazon.titan-text-lite-v1"


class BedrockClient:
    """Wrapper for AWS Bedrock runtime operations."""

    def __init__(self, region: str = REGION):
        self.bedrock       = boto3.client("bedrock-runtime", region_name=region)
        self.bedrock_agent = boto3.client("bedrock-agent-runtime", region_name=region)
        self.region        = region

    def invoke_claude(
        self,
        prompt:      str,
        model_id:    str  = CLAUDE_3_SONNET,
        max_tokens:  int  = 1024,
        temperature: float = 0.0,
        system:      Optional[str] = None,
    ) -> str:
        """Invoke Claude via Bedrock using the Messages API format."""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens":        max_tokens,
            "temperature":       temperature,
            "messages":          messages,
        }
        if system:
            body["system"] = system

        try:
            response = self.bedrock.invoke_model(
                modelId      = model_id,
                contentType  = "application/json",
                accept       = "application/json",
                body         = json.dumps(body),
            )
            result = json.loads(response["body"].read())
            return result["content"][0]["text"]
        except ClientError as e:
            logger.error(f"Bedrock InvokeModel error: {e}")
            raise

    def invoke_claude_streaming(
        self,
        prompt:    str,
        model_id:  str = CLAUDE_3_SONNET,
        max_tokens: int = 1024,
    ) -> Generator[str, None, None]:
        """Stream Claude response tokens as they arrive."""
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens":        max_tokens,
            "messages":          [{"role": "user", "content": prompt}],
        }
        response = self.bedrock.invoke_model_with_response_stream(
            modelId     = model_id,
            contentType = "application/json",
            accept      = "application/json",
            body        = json.dumps(body),
        )
        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk["type"] == "content_block_delta":
                yield chunk["delta"].get("text", "")

    def embed_text(self, text: str, model_id: str = TITAN_EMBED_V2) -> list[float]:
        """Generate text embedding using Amazon Titan Embed."""
        body = {"inputText": text[:8192]}  # Titan V2 max 8192 tokens
        try:
            response = self.bedrock.invoke_model(
                modelId     = model_id,
                contentType = "application/json",
                accept      = "application/json",
                body        = json.dumps(body),
            )
            result = json.loads(response["body"].read())
            return result["embedding"]
        except ClientError as e:
            logger.error(f"Titan Embed error: {e}")
            raise

    def retrieve_and_generate(
        self,
        query:              str,
        knowledge_base_id:  str,
        model_arn:          str = f"arn:aws:bedrock:{REGION}::foundation-model/{CLAUDE_3_SONNET}",
        max_results:        int = 5,
    ) -> dict:
        """
        Bedrock Knowledge Base: retrieve relevant docs and generate grounded response.
        Returns both the generated response and source attributions.
        """
        try:
            response = self.bedrock_agent.retrieve_and_generate(
                input={"text": query},
                retrieveAndGenerateConfiguration={
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": knowledge_base_id,
                        "modelArn":        model_arn,
                        "retrievalConfiguration": {
                            "vectorSearchConfiguration": {
                                "numberOfResults": max_results,
                            }
                        },
                    }
                }
            )
            return {
                "answer":   response["output"]["text"],
                "citations": response.get("citations", []),
                "session_id": response.get("sessionId"),
            }
        except ClientError as e:
            logger.error(f"Bedrock RAG error: {e}")
            raise

    def generate_clinical_narrative(
        self,
        patient_data: dict,
        task:         str = "discharge_summary",
    ) -> str:
        """Generate AI clinical narrative from structured patient data."""
        from src.bedrock.prompt_templates import build_clinical_prompt
        prompt = build_clinical_prompt(patient_data, task)
        return self.invoke_claude(
            prompt      = prompt,
            model_id    = CLAUDE_3_SONNET,
            max_tokens  = 512,
            temperature = 0.0,
            system      = "You are a clinical informatics assistant. Generate accurate, factual clinical text based only on the provided structured data. Do not add information not present in the input.",
        )
