# One image repository per service. Tags are immutable: a tag always means the same image, so a
# deployment can't change underneath you, and rolling back means pointing at the previous tag.

resource "aws_ecr_repository" "this" {
  for_each             = toset(["api", "pipeline", "mlflow"])
  name                 = "${var.name}-${each.key}"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = true # a demo: `terraform destroy` removes the images too

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "this" {
  for_each   = aws_ecr_repository.this
  repository = each.value.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the 10 most recent images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 10 }
      action       = { type = "expire" }
    }]
  })
}
