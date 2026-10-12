terraform {
  required_version = ">= 1.16.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "6.66.0" # pinned; released 2026-09-21
    }
    random = {
      source  = "hashicorp/random"
      version = "3.9.1"
    }
  }
  # State is local (terraform.tfstate, git-ignored): one person, short-lived demo environment.
  # A team would use an S3 backend with locking (`use_lockfile = true`) so state is shared and safe.
}

provider "aws" {
  region  = var.region
  profile = var.aws_profile

  default_tags {
    tags = {
      Project   = "customer-retention-ml-platform"
      ManagedBy = "terraform"
    }
  }
}

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  account_id = data.aws_caller_identity.current.account_id
  azs        = slice(data.aws_availability_zones.available.names, 0, 2)
}
