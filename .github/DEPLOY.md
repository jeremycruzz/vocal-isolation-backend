# Deployment (Backend) – EC2

GitHub Actions builds the Docker image, pushes to ECR, then SSHs to EC2 to pull and run the container on push to `main`.

## Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user access key (ECR push + optional SSH) |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `ECR_REGISTRY` | ECR registry URL (e.g. `123456789012.dkr.ecr.us-east-1.amazonaws.com`) |
| `ECR_REPOSITORY` | ECR repository name (e.g. `cap-backend`) |
| `SSH_PRIVATE_KEY` | Private key for SSH access to the EC2 instance |

## Required GitHub Variables

| Variable | Description |
|----------|-------------|
| `EC2_HOST` | EC2 public IP or hostname |
| `EC2_USER` | SSH user (e.g. `ubuntu`, `ec2-user`) |
| `AWS_REGION` | AWS region (e.g. `us-east-1`). Optional; defaults to `us-east-1`. |

## EC2 Setup

1. Launch an EC2 instance with Docker installed.
2. Attach an **IAM instance profile** with ECR pull permissions:
   - `ecr:GetAuthorizationToken`
   - `ecr:BatchGetImage`, `ecr:GetDownloadUrlForLayer` on your repository
3. Install AWS CLI v2 on the instance (for `aws ecr get-login-password`).
4. Add your SSH public key to the instance.
5. Open port 80 (and 443 for HTTPS) in the security group. Port 8000 does not need to be open to the internet when nginx on the same instance proxies to the backend.
6. **HTTPS (recommended):** If serving the site from this instance, add a domain and use Let's Encrypt (e.g. `certbot --nginx -d yourdomain.com`). See the frontend repo's `docs/AWS_SETUP_GUIDE.md` Part 5 (HTTPS with Let's Encrypt).
7. Add the secrets and variables in GitHub.

The workflow will run `docker pull` and `docker run` on the EC2 instance. Ensure checkpoints are in the image or mounted via a volume.
