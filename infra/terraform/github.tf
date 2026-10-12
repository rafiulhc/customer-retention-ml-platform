# GitHub Actions publishes images to ECR with short-lived credentials (OIDC): no AWS access keys are
# stored in GitHub. Only workflows from this repository's main branch can assume the role, and the
# role can only push to the three image repositories.

resource "aws_iam_openid_connect_provider" "github" {
  count          = var.create_github_oidc_provider ? 1 : 0
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
}

data "aws_iam_openid_connect_provider" "github" {
  count = var.create_github_oidc_provider ? 0 : 1
  url   = "https://token.actions.githubusercontent.com"
}

locals {
  github_owner    = split("/", var.github_repository)[0]
  github_repo     = split("/", var.github_repository)[1]
  github_oidc_arn = var.create_github_oidc_provider ? aws_iam_openid_connect_provider.github[0].arn : data.aws_iam_openid_connect_provider.github[0].arn
}

data "aws_iam_policy_document" "github_assume" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [local.github_oidc_arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    # GitHub's subject names the owner and repository with their permanent numeric IDs, e.g.
    # repo:owner@123/name@456:ref:refs/heads/main. If the repository were deleted and its name taken by
    # someone else, the IDs wouldn't match: the role can't be taken over by re-creating the name.
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${local.github_owner}@${var.github_ids.owner}/${local.github_repo}@${var.github_ids.repository}:ref:refs/heads/main"]
    }
  }
}

resource "aws_iam_role" "github_publish" {
  name               = "${var.name}-github-publish"
  assume_role_policy = data.aws_iam_policy_document.github_assume.json
}

resource "aws_iam_role_policy" "github_publish" {
  name = "push-images"
  role = aws_iam_role.github_publish.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = ["*"] # this action has no resource-level permissions
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability", "ecr:BatchGetImage", "ecr:CompleteLayerUpload",
          "ecr:DescribeImages", "ecr:InitiateLayerUpload", "ecr:PutImage", "ecr:UploadLayerPart",
        ]
        Resource = [for repo in aws_ecr_repository.this : repo.arn]
      },
    ]
  })
}
