resource "aws_ecs_cluster" "main" {
  name = var.name
}

resource "aws_cloudwatch_log_group" "this" {
  for_each          = toset(["api", "mlflow", "pipeline"])
  name              = "/ecs/${var.name}-${each.key}"
  retention_in_days = 14
}

# Private DNS inside the VPC: the API and the pipeline reach MLflow at http://mlflow.<name>.internal:5000
resource "aws_service_discovery_private_dns_namespace" "internal" {
  name = "${var.name}.internal"
  vpc  = aws_vpc.main.id
}

resource "aws_service_discovery_service" "mlflow" {
  name = "mlflow"
  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.internal.id
    dns_records {
      type = "A"
      ttl  = 10
    }
    routing_policy = "MULTIVALUE"
  }
}

locals {
  mlflow_uri = "http://mlflow.${aws_service_discovery_private_dns_namespace.internal.name}:5000"
  image      = { for k, repo in aws_ecr_repository.this : k => "${repo.repository_url}:${var.image_tags[k]}" }

  logging = { for k, group in aws_cloudwatch_log_group.this : k => {
    logDriver = "awslogs"
    options = {
      awslogs-group         = group.name
      awslogs-region        = data.aws_region.current.region
      awslogs-stream-prefix = k
    }
  } }
}

# ---------- MLflow server ----------
resource "aws_ecs_task_definition" "mlflow" {
  family                   = "${var.name}-mlflow"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.mlflow.arn
  runtime_platform {
    cpu_architecture        = "ARM64" # Graviton: ~20% cheaper; images are built for arm64
    operating_system_family = "LINUX"
  }

  container_definitions = jsonencode([{
    name         = "mlflow"
    image        = local.image["mlflow"]
    essential    = true
    portMappings = [{ containerPort = 5000 }]
    environment = [
      { name = "DB_HOST", value = aws_db_instance.mlflow.address },
      { name = "DB_NAME", value = aws_db_instance.mlflow.db_name },
      { name = "ARTIFACTS_DESTINATION", value = "s3://${aws_s3_bucket.this["artifacts"].bucket}" },
      # VPC addresses (health checks, internal clients), the private DNS name, and the UI on the load balancer
      { name = "ALLOWED_HOSTS", value = "10.0.*,mlflow.${aws_service_discovery_private_dns_namespace.internal.name}:*,${aws_lb.main.dns_name}:5001" },
      { name = "MLFLOW_SERVER_ENABLE_JOB_EXECUTION", value = "false" },
    ]
    secrets = [
      { name = "DB_USER", valueFrom = "${aws_db_instance.mlflow.master_user_secret[0].secret_arn}:username::" },
      { name = "DB_PASSWORD", valueFrom = "${aws_db_instance.mlflow.master_user_secret[0].secret_arn}:password::" },
    ]
    logConfiguration = local.logging["mlflow"]
  }])
}

resource "aws_ecs_service" "mlflow" {
  name            = "${var.name}-mlflow"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.mlflow.arn
  desired_count   = var.mlflow_desired_count
  launch_type     = "FARGATE"

  health_check_grace_period_seconds = 90

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.mlflow.id]
    assign_public_ip = true
  }
  service_registries {
    registry_arn = aws_service_discovery_service.mlflow.arn
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.mlflow.arn
    container_name   = "mlflow"
    container_port   = 5000
  }
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  depends_on = [aws_lb_listener.mlflow]
}

# ---------- API ----------
resource "aws_ecs_task_definition" "api" {
  family                   = "${var.name}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 256
  memory                   = 1024
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.api.arn
  runtime_platform {
    cpu_architecture        = "ARM64"
    operating_system_family = "LINUX"
  }

  container_definitions = jsonencode([{
    name             = "api"
    image            = local.image["api"]
    essential        = true
    portMappings     = [{ containerPort = 8000 }]
    environment      = [{ name = "MLFLOW_TRACKING_URI", value = local.mlflow_uri }]
    logConfiguration = local.logging["api"]
  }])
}

resource "aws_ecs_service" "api" {
  name            = "${var.name}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.api_desired_count
  launch_type     = "FARGATE"

  health_check_grace_period_seconds = 60

  # Rolling deploy: start the new version next to the old one, move traffic once it is healthy.
  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.api.id]
    assign_public_ip = true
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }

  # Automatic rollback, two ways: the new version never becomes healthy (circuit breaker), or it
  # becomes healthy but then errors or slows down (CloudWatch alarms watched during the deployment).
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  alarms {
    enable      = true
    rollback    = true
    alarm_names = [aws_cloudwatch_metric_alarm.api_5xx.alarm_name, aws_cloudwatch_metric_alarm.api_latency_p95.alarm_name]
  }
  depends_on = [aws_lb_listener.api]
}

# ---------- Pipeline tasks (run on a schedule, or by hand with `aws ecs run-task`) ----------
locals {
  pipeline_tasks = {
    retrain   = { command = ["python", "-m", "retention.pipeline.flows"], cpu = 1024, memory = 4096 }
    call-list = { command = ["python", "-m", "retention.pipeline.score"], cpu = 512, memory = 2048 }
  }
}

resource "aws_ecs_task_definition" "pipeline" {
  for_each                 = local.pipeline_tasks
  family                   = "${var.name}-${each.key}"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = each.value.cpu
  memory                   = each.value.memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.pipeline.arn
  runtime_platform {
    cpu_architecture        = "ARM64"
    operating_system_family = "LINUX"
  }

  container_definitions = jsonencode([{
    name      = each.key
    image     = local.image["pipeline"]
    essential = true
    command   = each.value.command
    environment = [
      { name = "MLFLOW_TRACKING_URI", value = local.mlflow_uri },
      { name = "CALL_LIST_S3_URI", value = "s3://${aws_s3_bucket.this["data"].bucket}/call_lists/" },
    ]
    logConfiguration = local.logging["pipeline"]
  }])
}
