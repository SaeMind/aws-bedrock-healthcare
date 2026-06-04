# AWS Bedrock Generative AI Healthcare Application

![Language](https://img.shields.io/badge/language-Python%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-AWS%20Bedrock%20%7C%20Lambda%20%7C%20CDK-orange)
![Models](https://img.shields.io/badge/models-Claude%203%20Sonnet%20%7C%20Titan-purple)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

> Serverless generative AI healthcare application on AWS using Bedrock as the LLM runtime. Provides RAG-powered clinical question answering over a CMS claims knowledge base, AI-generated patient risk narratives via Claude 3 Sonnet, and ICD-10 classification — all served through a Lambda-backed REST API provisioned entirely via AWS CDK.

---

## Key Capabilities

| Endpoint | Model | Latency (p50) | Use Case |
|----------|-------|--------------|----------|
| `POST /narrative` | Claude 3 Sonnet | ~1.4s | Discharge summary / risk narrative generation |
| `POST /rag/query` | Claude 3 Sonnet + KB | ~2.1s | Clinical Q&A grounded in CMS knowledge base |
| `POST /classify` | Claude 3 Haiku | ~0.6s | ICD-10 → clinical domain (12 categories) |
| `GET /health` | — | <50ms | Health check |

---

## Architecture

```
API Gateway (REST)
       │
       ▼
Lambda (Python 3.12, 512MB, 30s timeout)
       │
       ├── /narrative ──► Bedrock InvokeModel
       │                   Claude 3 Sonnet
       │                   clinical prompt templates
       │
       ├── /rag/query ──► Bedrock RetrieveAndGenerate
       │                   Knowledge Base (OpenSearch Serverless)
       │                   Titan Embed V2 (1536-dim vectors)
       │                   Claude 3 Sonnet synthesis
       │
       └── /classify ──► Bedrock InvokeModel
                          Claude 3 Haiku
                          JSON-mode structured output

Infrastructure (CDK):
  S3 bucket (CMS data + KB source docs)
  OpenSearch Serverless collection (vector search)
  IAM roles (least-privilege Bedrock + S3 access)
  CloudWatch log group (30-day retention)
  API Gateway + Lambda integration
```

---

## Sample Output

```bash
# Narrative generation
curl -X POST https://{api-id}.execute-api.us-east-1.amazonaws.com/v1/narrative \
  -H "Content-Type: application/json" \
  -d '{
    "task": "discharge_summary",
    "patient_data": {
      "age": 72, "admission_date": "2022-03-10", "discharge_date": "2022-03-15",
      "los_days": 5, "principal_dx": "I50.9", "drg_code": "291"
    }
  }'

{
  "narrative": "You were admitted to the hospital for 5 days for heart failure and
    treated with medications to remove excess fluid from your body. Your condition
    improved and you are being discharged home. Please follow up with your heart
    doctor within one week and weigh yourself every morning.",
  "task": "discharge_summary",
  "model": "claude-3-sonnet"
}

# ICD-10 classification
curl -X POST .../v1/classify \
  -d '{"icd_codes": ["I50.9", "E11.9", "N18.3"]}'

{
  "classification": {
    "primary_domain": "cardiovascular",
    "secondary_domain": "endocrine/metabolic",
    "confidence": 0.94
  }
}
```

---

## Bedrock Models

| Model ID | Tier | Use Case |
|----------|------|----------|
| `anthropic.claude-3-sonnet-20240229-v1:0` | Frontier | Narrative generation, RAG synthesis |
| `anthropic.claude-3-haiku-20240307-v1:0` | Fast/cheap | Classification, triage |
| `amazon.titan-embed-text-v2:0` | Embedding | Knowledge base document vectorization |

---

## Quick Start

```bash
git clone https://github.com/SaeMind/aws-bedrock-healthcare
cd aws-bedrock-healthcare

# Prerequisites: AWS CLI authenticated, CDK bootstrapped in target account/region
pip install -r infrastructure/lambda/requirements.txt
npm install -g aws-cdk

# Deploy infrastructure
cd infrastructure/cdk
cdk deploy HealthcareBedrockStack \
  --context account=YOUR_ACCOUNT_ID \
  --context region=us-east-1

# After deploy: create Bedrock Knowledge Base in console, then update Lambda env var:
# BEDROCK_KB_ID = <knowledge-base-id>
```

---

## Directory Structure

```
aws-bedrock-healthcare/
├── src/
│   ├── bedrock/
│   │   ├── bedrock_client.py       # InvokeModel, streaming, RetrieveAndGenerate, Titan Embed
│   │   └── prompt_templates.py     # discharge_summary, risk_narrative, classify, care_gap
│   └── api/
│       └── lambda_handler.py       # API Gateway router: /narrative /rag/query /classify /health
└── infrastructure/
    ├── cdk/healthcare_stack.py     # S3, OpenSearch Serverless, Lambda, API Gateway, IAM
    └── lambda/requirements.txt
```

---

## Citation

> Lee A. *AWS Bedrock generative AI healthcare application: serverless RAG and clinical narrative generation with Claude 3.* github.com/SaeMind/aws-bedrock-healthcare. 2024.

---

## Author

**Andrew Lee, MS** | Biomedical Informatics | UTHealth Houston SBMI
[LinkedIn](https://linkedin.com/in/agllee) · [Portfolio](https://andrew-gihbeom-lee.figma.site) · [GitHub](https://github.com/SaeMind)
