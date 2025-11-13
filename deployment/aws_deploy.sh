#!/bin/bash
# AWS Deployment Script for AgriSmart Application

echo "=== AgriSmart AWS Deployment Script ==="

# Configuration
APP_NAME="agrismart"
REGION="us-east-1"
ECR_REPO="agrismart-repo"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI is not installed. Please install it first."
    exit 1
fi

echo "Step 1: Building Docker image..."
docker build -t ${APP_NAME}:latest .

echo "Step 2: Logging into AWS ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

echo "Step 3: Creating ECR repository (if not exists)..."
aws ecr create-repository --repository-name ${ECR_REPO} --region ${REGION} || true

echo "Step 4: Tagging image..."
docker tag ${APP_NAME}:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:latest

echo "Step 5: Pushing image to ECR..."
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:latest

echo "Step 6: Deployment complete!"
echo "You can now deploy this image to:"
echo "  - AWS ECS (Elastic Container Service)"
echo "  - AWS EKS (Elastic Kubernetes Service)"
echo "  - AWS App Runner"
echo "  - AWS Elastic Beanstalk"

echo ""
echo "For ECS deployment, update the task definition with the new image URI:"
echo "${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:latest"
