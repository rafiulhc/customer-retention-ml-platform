# One load balancer, two doors:
#   :80   the API, only for requests carrying CloudFront's secret header (CloudFront adds HTTPS)
#   :5001 the MLflow UI, only from the operator's IP (security group). Plain HTTP: acceptable for a
#         short-lived demo restricted to one address; production would put it behind SSO or a VPN.

resource "random_password" "origin_verify" {
  length  = 32
  special = false
}

#trivy:ignore:AVD-AWS-0053 CloudFront's origin: only CloudFront's address ranges (and the operator, on :5001) can reach it
resource "aws_lb" "main" {
  name                       = "${var.name}-alb"
  load_balancer_type         = "application"
  drop_invalid_header_fields = true # reject malformed headers (request-smuggling tricks)
  subnets                    = aws_subnet.public[*].id
  security_groups            = [aws_security_group.alb.id]
}

resource "aws_lb_target_group" "api" {
  name                 = "${var.name}-api"
  port                 = 8000
  protocol             = "HTTP"
  target_type          = "ip"
  vpc_id               = aws_vpc.main.id
  deregistration_delay = 30
  health_check {
    path                = "/health"
    matcher             = "200"
    interval            = 15
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_target_group" "mlflow" {
  name                 = "${var.name}-mlflow"
  port                 = 5000
  protocol             = "HTTP"
  target_type          = "ip"
  vpc_id               = aws_vpc.main.id
  deregistration_delay = 30
  health_check {
    path    = "/health"
    matcher = "200"
  }
}

#trivy:ignore:AVD-AWS-0054 no domain/certificate: users get HTTPS from CloudFront; this hop is CloudFront -> ALB
resource "aws_lb_listener" "api" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"
  default_action {
    type = "fixed-response"
    fixed_response {
      content_type = "text/plain"
      message_body = "Forbidden"
      status_code  = "403"
    }
  }
}

resource "aws_lb_listener_rule" "from_cloudfront" {
  listener_arn = aws_lb_listener.api.arn
  priority     = 1
  condition {
    http_header {
      http_header_name = "X-Origin-Verify"
      values           = [random_password.origin_verify.result]
    }
  }
  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

#trivy:ignore:AVD-AWS-0054 demo only: restricted to the operator's IP; production would use SSO or a VPN
resource "aws_lb_listener" "mlflow" {
  load_balancer_arn = aws_lb.main.arn
  port              = 5001
  protocol          = "HTTP"
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.mlflow.arn
  }
}

# ---------- CloudFront: a free HTTPS address in front of the API ----------
data "aws_cloudfront_cache_policy" "disabled" {
  name = "Managed-CachingDisabled" # an API: never cache answers
}

data "aws_cloudfront_origin_request_policy" "all_viewer_except_host" {
  name = "Managed-AllViewerExceptHostHeader"
}

#trivy:ignore:AVD-AWS-0011 no WAF (~$5+/month): input is validated by the API's schema; add one before real traffic
resource "aws_cloudfront_distribution" "api" {
  enabled         = true
  comment         = "${var.name} API (HTTPS)"
  price_class     = "PriceClass_100"
  is_ipv6_enabled = true

  origin {
    origin_id   = "alb"
    domain_name = aws_lb.main.dns_name
    custom_header {
      name  = "X-Origin-Verify"
      value = random_password.origin_verify.result
    }
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "http-only" # no domain or certificate on the load balancer
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    target_origin_id         = "alb"
    viewer_protocol_policy   = "redirect-to-https"
    allowed_methods          = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods           = ["GET", "HEAD"]
    cache_policy_id          = data.aws_cloudfront_cache_policy.disabled.id
    origin_request_policy_id = data.aws_cloudfront_origin_request_policy.all_viewer_except_host.id
    compress                 = true
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  viewer_certificate {
    cloudfront_default_certificate = true
  }
}
