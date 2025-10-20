# Quick Start Guide - Deploy Training Without Building Docker

This guide shows you how to deploy your training job using a pre-built PyTorch image (no Docker build required!).

## What's Different?

Instead of building a custom Docker image, we:
1. Use the official PyTorch image (`pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`)
2. Clone your code from GitHub directly in the pod
3. Install dependencies at startup

**Advantages:**
- No need to build or push Docker images
- Faster iteration (just push code to GitHub)
- No local Docker setup required

## Prerequisites

1. **Kubernetes Access**: `kubectl` configured with your cluster credentials
2. **GitHub Repository**: Your code must be pushed to GitHub
3. **Persistent Volume**: PVC `v2v-diffuser-kuntal` should already exist

## Step 1: Verify Prerequisites

```bash
# Check kubectl access
kubectl get nodes

# Check PVC exists
kubectl get pvc v2v-diffuser-kuntal

# Should show: STATUS = Bound
```

## Step 2: Push Latest Code to GitHub

```bash
cd /Users/kuntalkokate/Desktop/LLM_agents_projects/LLM_agent_v2v

# Make sure you're on pretrained_main branch
git checkout pretrained_main

# Push any recent changes
git add .
git commit -m "Update configs for deployment"
git push origin pretrained_main
```

## Step 3: Deploy Interactive Pod (for Testing)

```bash
# Deploy the interactive pod
kubectl apply -f kub_files/interactive-pod.yaml

# Wait for it to be ready (this takes 3-5 minutes due to git clone + pip install)
kubectl get pod v2v-diffusion-interactive -w

# Watch the logs to see progress
kubectl logs -f v2v-diffusion-interactive
```

### Expected Output

You should see:
```
Installing system dependencies...
Cloning repository...
Installing Python dependencies...
Setup complete! Keeping container alive...
```

## Step 4: Access the Pod and Test

```bash
# Access the pod
kubectl exec -it v2v-diffusion-interactive -- /bin/bash

# Inside the pod:
cd /workspace/video-to-video-diffusion

# Check GPU
nvidia-smi

# Check Python packages
pip list | grep torch

# Test training script (dry run - just check it starts)
python train.py --config config/cloud_train_config.yaml
# Press Ctrl+C after you see it's working

# Exit the pod
exit
```

## Step 5: Clean Up Interactive Pod

```bash
# Delete the interactive pod
kubectl delete pod v2v-diffusion-interactive
```

## Step 6: Deploy Actual Training

Once you've verified everything works, deploy the training job:

### Option A: Deploy as a Job (Recommended)

```bash
# Deploy training job
kubectl apply -f kub_files/training-job.yaml

# Check job status
kubectl get job v2v-diffusion-training

# Get the pod name
kubectl get pods -l app=v2v-diffusion

# View logs
kubectl logs -f <pod-name>
# Example: kubectl logs -f v2v-diffusion-training-xxxxx
```

### Option B: Deploy as a Pod

```bash
# Deploy training pod
kubectl apply -f kub_files/training-pod.yaml

# Check status
kubectl get pod v2v-diffusion-training-pod

# View logs
kubectl logs -f v2v-diffusion-training-pod
```

## Step 7: Monitor Training

### View Real-time Logs

```bash
# For job
kubectl logs -f $(kubectl get pods -l app=v2v-diffusion,type=training -o jsonpath='{.items[0].metadata.name}')

# For pod
kubectl logs -f v2v-diffusion-training-pod
```

### Check GPU Usage

```bash
# Get pod name
POD_NAME=$(kubectl get pods -l app=v2v-diffusion,type=training -o jsonpath='{.items[0].metadata.name}')

# Exec into pod
kubectl exec -it $POD_NAME -- nvidia-smi

# Or watch continuously
kubectl exec -it $POD_NAME -- watch -n 1 nvidia-smi
```

### Access Training Outputs

Outputs are saved to `/workspace/storage/` on the persistent volume:

```bash
# Create temp pod to access storage
kubectl run temp-access --rm -it --image=ubuntu:latest --restart=Never \
  --overrides='
{
  "spec": {
    "containers": [{
      "name": "temp",
      "image": "ubuntu:latest",
      "stdin": true,
      "tty": true,
      "volumeMounts": [{
        "name": "storage",
        "mountPath": "/storage"
      }]
    }],
    "volumes": [{
      "name": "storage",
      "persistentVolumeClaim": {
        "claimName": "v2v-diffuser-kuntal"
      }
    }]
  }
}'

# Inside the temp pod:
ls -la /storage/
ls -la /storage/outputs/
ls -la /storage/checkpoints/
```

## Step 8: Update Code and Redeploy

If you need to update your code:

```bash
# 1. Update code locally
# 2. Push to GitHub
git add .
git commit -m "Update training code"
git push origin pretrained_main

# 3. Delete old job/pod
kubectl delete job v2v-diffusion-training
# OR
kubectl delete pod v2v-diffusion-training-pod

# 4. Redeploy
kubectl apply -f kub_files/training-job.yaml
# OR
kubectl apply -f kub_files/training-pod.yaml
```

The pod will automatically clone the latest code from GitHub!

## Troubleshooting

### Pod Stuck in "ContainerCreating"

```bash
# Check events
kubectl describe pod <pod-name>

# Common issues:
# - PVC not bound
# - Node selector doesn't match available nodes
```

### Pod Shows "Error" or "CrashLoopBackOff"

```bash
# Check logs
kubectl logs <pod-name>

# Common issues:
# - Git clone failed (repo private?)
# - Pip install failed (dependency issue?)
# - Out of memory
```

### Adjust Node Selector

If no GPU nodes match your selector, edit the YAML files:

```bash
# Check available GPU nodes
kubectl get nodes -L nvidia.com/gpu.product

# Option 1: Update nodeSelector in YAML to match
# Option 2: Remove nodeSelector entirely to use any GPU node
```

### Reduce Memory if Needed

Edit the YAML files to reduce memory requests:

```yaml
resources:
  requests:
    memory: "32Gi"  # Reduce from 48Gi
    cpu: "8"        # Reduce from 12
```

## Training Configuration

You can adjust training parameters by editing `config/cloud_train_config.yaml` in your repo:

- `batch_size`: Reduce if running out of memory
- `num_epochs`: Training duration
- `learning_rate`: Optimizer settings
- `checkpoint_every`: How often to save checkpoints

## Next Steps

1. Monitor training progress via logs
2. Check checkpoints in persistent storage
3. Adjust hyperparameters if needed
4. Run inference on trained model

## Summary

**What We Did:**
- ✅ Configured pods to use pre-built PyTorch image
- ✅ Set up automatic code cloning from GitHub
- ✅ Install dependencies at pod startup
- ✅ No Docker build required!

**To Deploy:**
1. Push code to GitHub
2. `kubectl apply -f kub_files/interactive-pod.yaml` (test)
3. `kubectl apply -f kub_files/training-job.yaml` (production)
4. Monitor with `kubectl logs -f <pod-name>`

That's it! Your training will run without needing to build any Docker images.
