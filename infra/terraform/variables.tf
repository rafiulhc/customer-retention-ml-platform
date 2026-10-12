variable "region" {
  type    = string
  default = "us-west-1"
}

variable "aws_profile" {
  type    = string
  default = "calldesk"
}

variable "name" {
  description = "Prefix for every resource name"
  type        = string
  default     = "retention"
}

variable "github_repository" {
  description = "owner/repo allowed to deploy through GitHub Actions (OIDC)"
  type        = string
  default     = "rafiulhc/customer-retention-ml-platform"
}

variable "github_ids" {
  description = "Permanent IDs of the owner and repository (gh api repos/OWNER/REPO --jq '.owner.id, .id')"
  type        = object({ owner = number, repository = number })
  default     = { owner = 68476971, repository = 1414432481 }
}

variable "operator_cidr" {
  description = "Your public IP as a /32: the only address allowed to open the MLflow UI"
  type        = string

  validation {
    condition     = can(cidrhost(var.operator_cidr, 0)) && endswith(var.operator_cidr, "/32")
    error_message = "Use a single address, e.g. 203.0.113.7/32."
  }
}

variable "image_tags" {
  description = "ECR tags to run (tags are immutable: use a new one per release)"
  type = object({
    api      = string
    pipeline = string
    mlflow   = string
  })
  default = { api = "v1", pipeline = "v1", mlflow = "v1" }
}

variable "mlflow_desired_count" {
  description = "Starts at 0: there is no image in ECR until the first push"
  type        = number
  default     = 0
}

variable "api_desired_count" {
  description = "Starts at 0: the API refuses to start until a champion has been trained and evaluated"
  type        = number
  default     = 0
}

variable "schedules_enabled" {
  description = "Run the monthly retrain and weekly call list on their schedules"
  type        = bool
  default     = false
}

variable "alarm_email" {
  description = "Email for alarms and the budget alert (empty: no notifications)"
  type        = string
  default     = ""
}

variable "monthly_budget_usd" {
  description = "Budget alert threshold for this project's resources"
  type        = number
  default     = 40
}

variable "create_github_oidc_provider" {
  description = "An AWS account has at most one GitHub OIDC provider: false reuses an existing one"
  type        = bool
  default     = true
}
