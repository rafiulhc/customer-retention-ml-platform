# Two buckets: MLflow artifacts (models, tables, figures) and data (DVC versions, call lists).
# Private, encrypted, no public access. force_destroy lets `terraform destroy` remove them with their
# contents: right for a demo environment, wrong for production data (which would also keep versions).

locals {
  buckets = {
    artifacts = "${var.name}-mlflow-artifacts-${local.account_id}"
    data      = "${var.name}-data-${local.account_id}"
  }
}

resource "aws_s3_bucket" "this" {
  for_each      = local.buckets
  bucket        = each.value
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "this" {
  for_each                = aws_s3_bucket.this
  bucket                  = each.value.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

#trivy:ignore:AVD-AWS-0132 AWS-managed encryption (SSE-S3); a customer-managed KMS key adds cost and key policy work
resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  for_each = aws_s3_bucket.this
  bucket   = each.value.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.this["data"].id
  versioning_configuration {
    status = "Enabled" # an overwritten call list or data file can be recovered
  }
}
