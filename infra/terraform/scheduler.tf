# EventBridge Scheduler starts the pipeline tasks: monthly retrain, weekly call list (UTC).
# Off by default (`schedules_enabled`), so a demo environment doesn't run jobs nobody asked for.

locals {
  schedules = {
    retrain   = "cron(0 6 1 * ? *)"   # 06:00 on the 1st of every month
    call-list = "cron(0 5 ? * MON *)" # 05:00 every Monday
  }
}

data "aws_iam_policy_document" "scheduler_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["scheduler.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }
  }
}

resource "aws_iam_role" "scheduler" {
  name               = "${var.name}-scheduler"
  assume_role_policy = data.aws_iam_policy_document.scheduler_assume.json
}

resource "aws_iam_role_policy" "scheduler_run_tasks" {
  name = "run-pipeline-tasks"
  role = aws_iam_role.scheduler.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ecs:RunTask"]
        Resource = [for td in aws_ecs_task_definition.pipeline : td.arn]
      },
      {
        Effect   = "Allow"
        Action   = ["iam:PassRole"]
        Resource = [aws_iam_role.execution.arn, aws_iam_role.pipeline.arn]
      },
    ]
  })
}

resource "aws_scheduler_schedule" "pipeline" {
  for_each                     = local.schedules
  name                         = "${var.name}-${each.key}"
  schedule_expression          = each.value
  schedule_expression_timezone = "UTC"
  state                        = var.schedules_enabled ? "ENABLED" : "DISABLED"

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = aws_ecs_cluster.main.arn
    role_arn = aws_iam_role.scheduler.arn
    ecs_parameters {
      task_definition_arn = aws_ecs_task_definition.pipeline[each.key].arn
      launch_type         = "FARGATE"
      network_configuration {
        subnets          = aws_subnet.public[*].id
        security_groups  = [aws_security_group.pipeline.id]
        assign_public_ip = true
      }
    }
    retry_policy {
      maximum_retry_attempts = 0 # the flows retry their own network steps; a failed run alerts instead
    }
  }
}
