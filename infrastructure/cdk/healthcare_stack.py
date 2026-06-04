"""
healthcare_stack.py
AWS CDK Stack — Bedrock Healthcare Application Infrastructure
Author: Andrew Lee | UTHealth Houston SBMI

Provisions:
  - S3 bucket (CMS data + KB source documents)
  - Bedrock Knowledge Base + OpenSearch Serverless collection
  - Lambda function (Python 3.12) + API Gateway REST endpoint
  - IAM roles and policies
  - CloudWatch log groups
"""

import aws_cdk as cdk
from aws_cdk import (
    aws_s3 as s3,
    aws_lambda as lambda_,
    aws_apigateway as apigw,
    aws_iam as iam,
    aws_logs as logs,
    aws_opensearchserverless as oss,
    Duration, RemovalPolicy, Stack, CfnOutput,
)
from constructs import Construct


class HealthcareBedrockStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        # ─── S3 Bucket: CMS data + KB source documents ────────────────────────
        data_bucket = s3.Bucket(
            self, "HealthcareDataBucket",
            bucket_name     = f"healthcare-bedrock-{self.account}-{self.region}",
            removal_policy  = RemovalPolicy.RETAIN,
            versioned        = True,
            encryption       = s3.BucketEncryption.S3_MANAGED,
            block_public_access = s3.BlockPublicAccess.BLOCK_ALL,
        )

        # ─── OpenSearch Serverless — KB vector store ──────────────────────────
        oss_collection = oss.CfnCollection(
            self, "HealthcareKBCollection",
            name  = "healthcare-kb-vectors",
            type  = "VECTORSEARCH",
        )

        # ─── IAM Role: Bedrock Knowledge Base ─────────────────────────────────
        kb_role = iam.Role(
            self, "BedrockKBRole",
            assumed_by    = iam.ServicePrincipal("bedrock.amazonaws.com"),
            description   = "Role for Bedrock Knowledge Base to access S3 and OpenSearch",
        )
        kb_role.add_to_policy(iam.PolicyStatement(
            actions   = ["s3:GetObject", "s3:ListBucket"],
            resources = [data_bucket.bucket_arn, f"{data_bucket.bucket_arn}/*"],
        ))
        kb_role.add_to_policy(iam.PolicyStatement(
            actions   = ["aoss:APIAccessAll"],
            resources = [oss_collection.attr_arn],
        ))

        # ─── IAM Role: Lambda ─────────────────────────────────────────────────
        lambda_role = iam.Role(
            self, "HealthcareLambdaRole",
            assumed_by = iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies = [
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole"),
            ],
        )
        lambda_role.add_to_policy(iam.PolicyStatement(
            actions   = [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
                "bedrock:Retrieve",
                "bedrock:RetrieveAndGenerate",
            ],
            resources = ["*"],
        ))

        # ─── Lambda Function ──────────────────────────────────────────────────
        log_group = logs.LogGroup(
            self, "LambdaLogGroup",
            log_group_name  = "/aws/lambda/healthcare-bedrock-api",
            retention       = logs.RetentionDays.ONE_MONTH,
            removal_policy  = RemovalPolicy.DESTROY,
        )

        handler = lambda_.Function(
            self, "HealthcareBedrockLambda",
            function_name   = "healthcare-bedrock-api",
            runtime         = lambda_.Runtime.PYTHON_3_12,
            code            = lambda_.Code.from_asset("src"),
            handler         = "api.lambda_handler.lambda_handler",
            role            = lambda_role,
            memory_size     = 512,
            timeout         = Duration.seconds(30),
            environment     = {
                "AWS_REGION":   self.region,
                # BEDROCK_KB_ID injected post-KB creation
            },
            log_group       = log_group,
        )

        # ─── API Gateway ──────────────────────────────────────────────────────
        api = apigw.RestApi(
            self, "HealthcareBedrockApi",
            rest_api_name   = "healthcare-bedrock-api",
            description     = "Bedrock-powered clinical AI REST API",
            default_cors_preflight_options = apigw.CorsOptions(
                allow_origins = apigw.Cors.ALL_ORIGINS,
                allow_methods = apigw.Cors.ALL_METHODS,
            ),
            deploy_options  = apigw.StageOptions(
                stage_name    = "v1",
                logging_level = apigw.MethodLoggingLevel.INFO,
            ),
        )

        integration = apigw.LambdaIntegration(handler)
        health      = api.root.add_resource("health")
        health.add_method("GET", integration)

        narrative = api.root.add_resource("narrative")
        narrative.add_method("POST", integration)

        rag      = api.root.add_resource("rag")
        rag_query = rag.add_resource("query")
        rag_query.add_method("POST", integration)

        classify = api.root.add_resource("classify")
        classify.add_method("POST", integration)

        # ─── Outputs ─────────────────────────────────────────────────────────
        CfnOutput(self, "ApiUrl",
                  value=api.url,
                  description="Bedrock Healthcare API Gateway URL")
        CfnOutput(self, "DataBucketName",
                  value=data_bucket.bucket_name,
                  description="S3 bucket for CMS data and KB documents")
        CfnOutput(self, "KBRoleArn",
                  value=kb_role.role_arn,
                  description="IAM role ARN for Bedrock Knowledge Base")


app = cdk.App()
HealthcareBedrockStack(
    app, "HealthcareBedrockStack",
    env=cdk.Environment(
        account=app.node.try_get_context("account"),
        region=app.node.try_get_context("region") or "us-east-1",
    ),
)
app.synth()
