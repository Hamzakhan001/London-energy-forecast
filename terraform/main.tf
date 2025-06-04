# terraform/simplified_main.tf
# Simplified Terraform that works with existing buckets and limited permissions

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "eu-west-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "energy-ml"
}

# Data source for current AWS account
data "aws_caller_identity" "current" {}

# Data sources for existing S3 buckets (instead of creating new ones)
data "aws_s3_bucket" "raw_data" {
  bucket = "${var.project_name}-raw-data-${data.aws_caller_identity.current.account_id}"
}

data "aws_s3_bucket" "processed_data" {
  bucket = "${var.project_name}-processed-data-${data.aws_caller_identity.current.account_id}"
}

data "aws_s3_bucket" "features" {
  bucket = "${var.project_name}-features-${data.aws_caller_identity.current.account_id}"
}

data "aws_s3_bucket" "models" {
  bucket = "${var.project_name}-models-${data.aws_caller_identity.current.account_id}"
}

data "aws_s3_bucket" "predictions" {
  bucket = "${var.project_name}-predictions-${data.aws_caller_identity.current.account_id}"
}

# S3 Bucket Versioning for existing buckets
resource "aws_s3_bucket_versioning" "buckets_versioning" {
  for_each = {
    raw_data       = data.aws_s3_bucket.raw_data.id
    processed_data = data.aws_s3_bucket.processed_data.id
    features       = data.aws_s3_bucket.features.id
    models         = data.aws_s3_bucket.models.id
    predictions    = data.aws_s3_bucket.predictions.id
  }
  
  bucket = each.value
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 Bucket Lifecycle Configuration (fixed to avoid warnings)
resource "aws_s3_bucket_lifecycle_configuration" "lifecycle" {
  for_each = {
    raw_data       = data.aws_s3_bucket.raw_data.id
    processed_data = data.aws_s3_bucket.processed_data.id
    features       = data.aws_s3_bucket.features.id
    models         = data.aws_s3_bucket.models.id
    predictions    = data.aws_s3_bucket.predictions.id
  }
  
  bucket = each.value

  rule {
    id     = "transition_to_ia"
    status = "Enabled"
    
    # Add filter to fix the warning
    filter {
      prefix = ""
    }

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
    
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
  }
}

# CloudWatch Log Groups (these usually work with basic permissions)
resource "aws_cloudwatch_log_group" "ml_pipeline_logs" {
  name              = "/aws/energy-ml/pipeline-logs"
  retention_in_days = 14

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

# SSM Parameters to store configuration (alternative to environment variables)
resource "aws_ssm_parameter" "raw_data_bucket" {
  name  = "/${var.project_name}/config/raw-data-bucket"
  type  = "String"
  value = data.aws_s3_bucket.raw_data.bucket

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_ssm_parameter" "processed_data_bucket" {
  name  = "/${var.project_name}/config/processed-data-bucket"
  type  = "String"
  value = data.aws_s3_bucket.processed_data.bucket

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_ssm_parameter" "features_bucket" {
  name  = "/${var.project_name}/config/features-bucket"
  type  = "String"
  value = data.aws_s3_bucket.features.bucket

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_ssm_parameter" "models_bucket" {
  name  = "/${var.project_name}/config/models-bucket"
  type  = "String"
  value = data.aws_s3_bucket.models.bucket

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_ssm_parameter" "predictions_bucket" {
  name  = "/${var.project_name}/config/predictions-bucket"
  type  = "String"
  value = data.aws_s3_bucket.predictions.bucket

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

# Outputs
output "s3_buckets" {
  description = "S3 bucket names"
  value = {
    raw_data       = data.aws_s3_bucket.raw_data.bucket
    processed_data = data.aws_s3_bucket.processed_data.bucket
    features       = data.aws_s3_bucket.features.bucket
    models         = data.aws_s3_bucket.models.bucket
    predictions    = data.aws_s3_bucket.predictions.bucket
  }
}

output "ssm_parameters" {
  description = "SSM parameter names for configuration"
  value = {
    raw_data_bucket       = aws_ssm_parameter.raw_data_bucket.name
    processed_data_bucket = aws_ssm_parameter.processed_data_bucket.name
    features_bucket       = aws_ssm_parameter.features_bucket.name
    models_bucket         = aws_ssm_parameter.models_bucket.name
    predictions_bucket    = aws_ssm_parameter.predictions_bucket.name
  }
}

output "account_id" {
  description = "AWS Account ID"
  value       = data.aws_caller_identity.current.account_id
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group for ML pipeline"
  value       = aws_cloudwatch_log_group.ml_pipeline_logs.name
}