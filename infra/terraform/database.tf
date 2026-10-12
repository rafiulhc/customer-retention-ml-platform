# PostgreSQL for MLflow's runs and model registry, in the private subnets (no internet route).
# The master password is generated and stored by RDS in Secrets Manager: it never appears in
# Terraform's code, variables or outputs.

resource "aws_db_subnet_group" "main" {
  name       = "${var.name}-db"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_db_instance" "mlflow" {
  identifier     = "${var.name}-mlflow"
  engine         = "postgres"
  engine_version = "17"
  instance_class = "db.t4g.micro"

  allocated_storage = 20
  storage_type      = "gp3"
  storage_encrypted = true

  db_name                     = "mlflow"
  username                    = "mlflow"
  manage_master_user_password = true

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.db.id]
  publicly_accessible    = false
  multi_az               = false # one AZ: a demo; production would use Multi-AZ

  backup_retention_period = 1
  skip_final_snapshot     = true # a demo: `terraform destroy` removes it without a snapshot
  deletion_protection     = false
  apply_immediately       = true
}
