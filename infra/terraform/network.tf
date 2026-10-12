# VPC with 2 public subnets (load balancer + containers) and 2 private subnets (database).
# No NAT gateway (~$33/month): containers run in public subnets with public IPs to pull images and
# reach AWS APIs, and security groups decide who can talk to whom.

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = { Name = "${var.name}-vpc" }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${var.name}-igw" }
}

#trivy:ignore:AVD-AWS-0164 by design: no NAT gateway, so containers need public IPs (inbound is closed by security groups)
resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.${count.index + 1}.0/24"
  availability_zone       = local.azs[count.index]
  map_public_ip_on_launch = true
  tags                    = { Name = "${var.name}-public-${local.azs[count.index]}" }
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 11}.0/24"
  availability_zone = local.azs[count.index]
  tags              = { Name = "${var.name}-private-${local.azs[count.index]}" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  tags = { Name = "${var.name}-public" }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# ---------- Security groups: each layer accepts traffic only from the layer in front ----------
data "aws_ec2_managed_prefix_list" "cloudfront" {
  name = "com.amazonaws.global.cloudfront.origin-facing"
}

resource "aws_security_group" "alb" {
  name        = "${var.name}-alb"
  description = "Load balancer: the API from CloudFront, the MLflow UI from the operator only"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "API traffic from CloudFront edge locations"
    from_port       = 80
    to_port         = 80
    protocol        = "tcp"
    prefix_list_ids = [data.aws_ec2_managed_prefix_list.cloudfront.id]
  }
  ingress {
    description = "MLflow UI from the operator IP"
    from_port   = 5001
    to_port     = 5001
    protocol    = "tcp"
    cidr_blocks = [var.operator_cidr]
  }
  egress {
    description = "To the containers"
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.main.cidr_block]
  }
}

resource "aws_security_group" "api" {
  name        = "${var.name}-api"
  description = "API containers: port 8000 only from the load balancer"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "From the load balancer"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  #trivy:ignore:AVD-AWS-0104 HTTPS to AWS APIs (ECR, CloudWatch) without a NAT gateway or VPC endpoints
  egress {
    description = "HTTPS out (ECR, CloudWatch)"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    description = "MLflow, inside the VPC"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.main.cidr_block]
  }
}

resource "aws_security_group" "pipeline" {
  name        = "${var.name}-pipeline"
  description = "Pipeline tasks: no inbound traffic at all"
  vpc_id      = aws_vpc.main.id

  #trivy:ignore:AVD-AWS-0104 HTTPS to the dataset source and AWS APIs without a NAT gateway or VPC endpoints
  egress {
    description = "HTTPS out (dataset download, ECR, S3, CloudWatch)"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    description = "MLflow, inside the VPC"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.main.cidr_block]
  }
}

resource "aws_security_group" "mlflow" {
  name        = "${var.name}-mlflow"
  description = "MLflow server: port 5000 from the API, the pipeline and the load balancer"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "From the API, the pipeline and the load balancer (UI)"
    from_port       = 5000
    to_port         = 5000
    protocol        = "tcp"
    security_groups = [aws_security_group.api.id, aws_security_group.pipeline.id, aws_security_group.alb.id]
  }
  #trivy:ignore:AVD-AWS-0104 HTTPS to AWS APIs (S3, ECR, Secrets Manager, CloudWatch) without a NAT gateway
  egress {
    description = "HTTPS out (S3, ECR, Secrets Manager, CloudWatch)"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    description = "Postgres in the private subnets"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [for s in aws_subnet.private : s.cidr_block]
  }
}

resource "aws_security_group" "db" {
  name        = "${var.name}-db"
  description = "Database: Postgres only from the MLflow server"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "From the MLflow server"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.mlflow.id]
  }
}
