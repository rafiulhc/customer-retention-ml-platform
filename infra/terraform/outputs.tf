output "api_url" {
  description = "The API over HTTPS (CloudFront)"
  value       = "https://${aws_cloudfront_distribution.api.domain_name}"
}

output "mlflow_ui" {
  description = "The MLflow UI (only from operator_cidr)"
  value       = "http://${aws_lb.main.dns_name}:5001"
}

output "ecr_repositories" {
  value = { for k, repo in aws_ecr_repository.this : k => repo.repository_url }
}

output "buckets" {
  value = { for k, b in aws_s3_bucket.this : k => b.bucket }
}

output "github_publish_role_arn" {
  description = "Set as the AWS_PUBLISH_ROLE_ARN variable of the GitHub repository"
  value       = aws_iam_role.github_publish.arn
}

output "run_task" {
  description = "Settings for `aws ecs run-task` (start a pipeline task by hand)"
  value = {
    cluster        = aws_ecs_cluster.main.name
    subnets        = aws_subnet.public[*].id
    security_group = aws_security_group.pipeline.id
    task_families  = [for td in aws_ecs_task_definition.pipeline : td.family]
  }
}
