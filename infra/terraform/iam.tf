# One role per job, each with only what that job needs (least privilege).
#   execution role:  what ECS itself needs to start a container (pull the image, write logs, read the DB secret)
#   task roles:      what the code inside the container may do

data "aws_iam_policy_document" "ecs_tasks_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${var.name}-ecs-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "execution_db_secret" {
  name = "read-db-secret"
  role = aws_iam_role.execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = [aws_db_instance.mlflow.master_user_secret[0].secret_arn]
    }]
  })
}

# MLflow: read and write model artifacts. The API and the pipeline never touch this bucket
# directly: they go through MLflow (--serve-artifacts), so they need no storage permissions.
resource "aws_iam_role" "mlflow" {
  name               = "${var.name}-mlflow-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

resource "aws_iam_role_policy" "mlflow_artifacts" {
  name = "artifacts-bucket"
  role = aws_iam_role.mlflow.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = [aws_s3_bucket.this["artifacts"].arn]
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
        Resource = ["${aws_s3_bucket.this["artifacts"].arn}/*"]
      },
    ]
  })
}

# The API: no AWS permissions at all (it reads the model through MLflow).
resource "aws_iam_role" "api" {
  name               = "${var.name}-api-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

# The pipeline: write call lists, nothing else.
resource "aws_iam_role" "pipeline" {
  name               = "${var.name}-pipeline-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

resource "aws_iam_role_policy" "pipeline_call_lists" {
  name = "write-call-lists"
  role = aws_iam_role.pipeline.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:PutObject"]
      Resource = ["${aws_s3_bucket.this["data"].arn}/call_lists/*"]
    }]
  })
}
