# Deployment on AWS (module 9)

Everything is code: `infra/terraform` (infrastructure), three Dockerfiles (images), `.github/workflows/publish.yml`
(image publishing with OIDC), `dvc.yaml` (data versions). The environment is built for **short demos**: it is
created, used, and destroyed, so it costs money only while it exists.

```
                     CloudFront (HTTPS) ──▶ ALB :80 ──▶ API service (ECS Fargate, rolling deploy, auto-rollback)
operator (one IP) ──▶ ALB :5001 ───────────────────▶ MLflow server (ECS) ──▶ RDS Postgres (private subnets)
                                                          ▲              └─▶ S3: model artifacts
EventBridge Scheduler ──▶ pipeline tasks (ECS) ───────────┘   retrain (monthly), call list (weekly) ──▶ S3
GitHub Actions ──OIDC (no stored keys)──▶ ECR images          DVC ──▶ S3: data versions
```

## What gets created

| Area | Resources | Notes |
|---|---|---|
| Network | VPC, 2 public + 2 private subnets, security groups per layer | **No NAT gateway** (saves ~$33/month): containers use public IPs, inbound closed by security groups |
| Compute | ECS cluster on **Graviton (arm64)**: MLflow (0.5 vCPU / 1 GB), API (0.25 vCPU / 1 GB), pipeline tasks on demand | arm64: ~20% cheaper than x86 |
| Data | RDS Postgres 17 (db.t4g.micro, private), S3 artifacts + data buckets | DB password generated and kept by RDS in Secrets Manager |
| Entry | ALB + CloudFront (HTTPS without a domain), MLflow UI on :5001 for your IP only | |
| Releases | ECR (immutable tags, scan on push), GitHub OIDC role (push images only, main branch only) | |
| Operations | EventBridge schedules (off by default), CloudWatch logs and alarms, pipeline-failure alert, budget alert | |

## Safe deployments

The API service uses a **rolling deploy**: new containers start next to the old ones, and traffic moves once they
pass the health check. Rollback is automatic in two cases:

- **Circuit breaker:** the new version never becomes healthy, e.g. it can't load a model. ECS stops the deployment
  and puts the previous version back.
- **Deployment alarms:** the new version is healthy but then returns errors (5xx) or slows down (p95 > 200 ms).
  ECS rolls back.

Model rollback is separate and faster: move `@champion` back and restart the service.

## Cost (us-west-1, on-demand, estimated)

| Item | While fully running | Per month |
|---|---:|---:|
| ALB | $0.025/h | ~$20 |
| RDS db.t4g.micro + 20 GB | $0.023/h | ~$17 |
| Fargate: MLflow + API (arm64) | $0.036/h | ~$26 |
| Public IPv4 addresses (4) | $0.020/h | ~$15 |
| Secrets Manager, private DNS zone, S3, ECR, CloudWatch, CloudFront | | ~$3 |
| Pipeline tasks (minutes per month) | | <$1 |
| **Total** | **~$0.11/hour** | **~$80** |

A demo session of a few hours costs **about $1–3**. With the services at 0 containers, the fixed part (ALB, RDS,
IPv4) is about $1.70/day. **Destroy the environment after the demo.**

## Runbook

```bash
aws sso login --profile calldesk
cd infra/terraform && cp terraform.tfvars.example terraform.tfvars   # set operator_cidr (and alarm_email)
terraform init && terraform plan -out tfplan && terraform apply tfplan  # phase 1: everything, 0 containers
```

1. **Images:** push `api`, `pipeline` and `mlflow` to ECR with one tag, either from a Mac (native arm64) or with the
   *Publish images* workflow (repository variable `AWS_PUBLISH_ROLE_ARN` = the `github_publish_role_arn` output).
2. **MLflow:** `terraform apply -var mlflow_desired_count=1`, then open the `mlflow_ui` output.
3. **First champion, trained in the cloud:** start the `retention-retrain` task (`aws ecs run-task`, settings in
   the `run_task` output). The pipeline downloads, validates, trains, verifies, evaluates once and promotes through
   the gate.
4. **API:** `terraform apply -var api_desired_count=1`, then `POST <api_url>/predict`.
5. **Call list:** start the `retention-call-list` task. The list lands in the data bucket under `call_lists/`.
6. **Data versions:** `dvc remote add -d s3 s3://<data bucket>/dvc && dvc push`.
7. **Destroy:** `terraform destroy`.

## Limitations (deliberate, for a demo)

- Local Terraform state: a team would use an S3 backend with locking.
- CloudFront → ALB and the MLflow UI use plain HTTP (no domain or certificate). MLflow is restricted to one IP;
  production would put it behind SSO or a VPN.
- One NAT-less VPC, single-AZ database, no WAF, AWS-managed encryption keys. Each is annotated in the code where a
  security scanner would flag it.

## First deployment (2026-10-12): what happened

| Step | Result |
|---|---|
| `terraform apply` | 71 resources (one retry: see below) |
| Images | api 224 MB, mlflow 308 MB, pipeline 399 MB (compressed, arm64) |
| First champion, trained **in the cloud** | fetch → validate → train → register → verify → evaluate → gate → promote in **18 s**. Same data fingerprint, test PR-AUC (0.6315), profit ($19,678) and threshold (0.28) as on the laptop; lineage `dbb572f`, clean; passed criteria version 2 |
| API over HTTPS | same prediction as locally (0.7717 for the example customer). **In-container p95 78 ms** (0.25 vCPU; requirement 200 ms). The ~0.8 s round trip from the laptop is internet distance to us-west-1. Direct requests to the load balancer time out: only CloudFront is let in |
| Weekly call list | 5,174 scored, 1,311 worth calling, 100 listed, ~22 expected saves. The file was written to S3 |
| DVC | 4 files pushed. A deleted training file was pulled back byte-identical |
| **Automatic rollback** | a broken release (MLflow unreachable) was deployed at 12:57:14. ECS reported "tasks failed to start" and rolled back to the previous revision by 13:03:54, **with no one involved**. **40/40** health checks passed during the attempt |
| Drift | `terraform plan` afterwards: no changes |

### Issues found only by running it for real

1. **AWS rejects apostrophes in security group descriptions.** `terraform validate` can't know that; `plan` found it.
2. **SNS topic policies need a `Sid` per statement.** Apply failed on that one resource. It was fixed and re-applied.
3. **The pipeline image crashed when registering:** the model's requirements pinned `mlflow`, which only the dev
   environment has (images have `mlflow-skinny`). There is now a test that every runtime requirement is a core dependency.
4. **Lineage was lost:** images have no git, so commits came out as "unknown". The build now records the commit
   (`GIT_COMMIT` build argument).
5. **Model downloads failed on AWS too:** S3 rejected MLflow's presigned links (`SignatureDoesNotMatch` with
   task-role credentials, reproduced from outside the VPC). Clients now stream through the MLflow server, as
   locally. The model registered during that failed run was **never promoted**: verification stopped the run first.
6. **A false alarm:** on Linux, XGBoost edged out logistic regression by a hair, a tie. The "revisit the model
   choice" warning now needs a gap beyond 0.005 PR-AUC.
