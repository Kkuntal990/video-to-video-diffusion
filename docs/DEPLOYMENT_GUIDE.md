# Kubernetes Deployment Guide for CT Slice Interpolation Training

This guide walks you through deploying and training your CT slice interpolation model using latent diffusion on a Kubernetes cluster (Nautilus/NRP).

## Prerequisites

1. **Kubernetes Access**: Ensure you have access to a Kubernetes cluster with GPU nodes
2. **kubectl**: Install and configure kubectl with your cluster credentials
3. **Docker**: Docker installed locally for building images
4. **Docker Registry Access**: Access to push images (Docker Hub, GitHub Container Registry, etc.)

## Step 1: Build and Push Docker Image

### 1.1 Build the Docker Image

```bash
# Navigate to project root
cd /Users/kuntalkokate/Desktop/LLM_agents_projects/LLM_agent_v2v

# Build the Docker image (replace with your Docker Hub username)
docker build -t <your-dockerhub-username>/v2v-diffusion:latest .

# Example:
# docker build -t kkuntal990/v2v-diffusion:latest .
```

### 1.2 Test the Docker Image Locally (Optional)

```bash
# Test the image locally
docker run --rm -it --gpus all <your-dockerhub-username>/v2v-diffusion:latest python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 1.3 Push to Docker Registry

```bash
# Login to Docker Hub
docker login

# Push the image
docker push <your-dockerhub-username>/v2v-diffusion:latest
```

## Step 2: Set Up Kubernetes Resources

### 2.1 Create Persistent Volume Claim (Already Done)

Your PVC is already configured in `kub_files/persistent_storage.yaml`. If not created yet:

```bash
kubectl apply -f kub_files/persistent_storage.yaml
```

Verify the PVC is created:
```bash
kubectl get pvc v2v-diffuser-kuntal
```

### 2.2 Update Kubernetes YAML Files

Edit the YAML files to replace `<YOUR_DOCKER_IMAGE>` with your actual image:

```bash
# In all YAML files, replace:
# <YOUR_DOCKER_IMAGE>
# with:
# <your-dockerhub-username>/v2v-diffusion:latest

# For example:
sed -i 's|<YOUR_DOCKER_IMAGE>|kkuntal990/v2v-diffusion:latest|g' kub_files/interactive-pod.yaml
sed -i 's|<YOUR_DOCKER_IMAGE>|kkuntal990/v2v-diffusion:latest|g' kub_files/training-pod.yaml
sed -i 's|<YOUR_DOCKER_IMAGE>|kkuntal990/v2v-diffusion:latest|g' kub_files/training-job.yaml
```

### 2.3 Adjust Node Selector (Optional)

Check available GPU types in your cluster:
```bash
kubectl get nodes -L nvidia.com/gpu.product
```

Update the `nodeSelector` in YAML files if needed, or remove it to use any available GPU node.

## Step 3: Deploy Interactive Pod for Testing

### 3.1 Create the Interactive Pod

```bash
kubectl apply -f kub_files/interactive-pod.yaml
```

### 3.2 Wait for Pod to be Running

```bash
kubectl get pod v2v-diffusion-interactive -w
```

Wait until STATUS shows "Running".

### 3.3 Access the Interactive Pod

```bash
kubectl exec -it v2v-diffusion-interactive -- /bin/bash
```

### 3.4 Verify Environment Inside Pod

Once inside the pod, run these verification commands:

```bash
# Check GPU availability
nvidia-smi

# Check Python and PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"

# Check if code is accessible
ls -la /workspace

# Check storage mount
ls -la /workspace/storage

# Test data loading (quick test)
python -c "
from datasets import load_dataset
print('Testing HuggingFace dataset access...')
ds = load_dataset('t2ance/APE-data', split='train', streaming=True)
print('Dataset loaded successfully!')
sample = next(iter(ds))
print(f'Sample keys: {sample.keys()}')
"
```

### 3.5 Test Training Command (Dry Run)

```bash
# Inside the pod, try running training for a few steps
cd /workspace && python -m training.train --config config/cloud_train_config.yaml
```

**Note**: Monitor the output for any errors. Press Ctrl+C after confirming it starts successfully.

### 3.6 Exit and Delete Interactive Pod

```bash
# Exit the pod
exit

# Delete the interactive pod
kubectl delete pod v2v-diffusion-interactive
```

## Step 4: Deploy Training Job

Once you've verified everything works in the interactive pod, deploy the actual training job.

### Option A: Deploy as a Job (Recommended)

Jobs are better for long-running training as they handle retries and completion tracking.

```bash
kubectl apply -f kub_files/training-job.yaml
```

Monitor the job:
```bash
# Check job status
kubectl get job v2v-diffusion-training

# Check pod status
kubectl get pods -l app=v2v-diffusion

# View logs (replace pod name)
kubectl logs -f <pod-name>

# Example:
# kubectl logs -f v2v-diffusion-training-xxxxx
```

### Option B: Deploy as a Pod

```bash
kubectl apply -f kub_files/training-pod.yaml
```

Monitor the pod:
```bash
# Check pod status
kubectl get pod v2v-diffusion-training-pod

# View logs
kubectl logs -f v2v-diffusion-training-pod
```

## Step 5: Monitor Training

### 5.1 View Real-time Logs

```bash
# For Job
kubectl logs -f $(kubectl get pods -l app=v2v-diffusion -o jsonpath='{.items[0].metadata.name}')

# For Pod
kubectl logs -f v2v-diffusion-training-pod
```

### 5.2 Check GPU Usage

```bash
# Exec into the running training pod
kubectl exec -it <pod-name> -- nvidia-smi

# Watch GPU usage continuously
kubectl exec -it <pod-name> -- watch -n 1 nvidia-smi
```

### 5.3 Access Training Outputs

The training outputs (checkpoints, logs) are saved to the persistent volume at `/workspace/storage/`.

To access them:

```bash
# Create a temporary pod to access storage
kubectl run storage-access --rm -it --image=ubuntu:latest --restart=Never \
  --overrides='
{
  "spec": {
    "containers": [{
      "name": "storage-access",
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

# Inside the storage-access pod:
ls -la /storage/
ls -la /storage/outputs/
ls -la /storage/checkpoints/
```

### 5.4 Copy Files from Persistent Storage (Optional)

```bash
# Create a temporary access pod
kubectl run temp-access --image=ubuntu --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"temp","image":"ubuntu","command":["/bin/bash","-c","sleep 3600"],"volumeMounts":[{"name":"storage","mountPath":"/storage"}]}],"volumes":[{"name":"storage","persistentVolumeClaim":{"claimName":"v2v-diffuser-kuntal"}}]}}'

# Wait for pod to be ready
kubectl wait --for=condition=Ready pod/temp-access --timeout=60s

# Copy files from storage to local machine
kubectl cp temp-access:/storage/outputs ./local_outputs
kubectl cp temp-access:/storage/checkpoints ./local_checkpoints

# Delete temporary pod
kubectl delete pod temp-access
```

## Step 6: TensorBoard Monitoring (Optional)

### 6.1 Create TensorBoard Pod

Create `kub_files/tensorboard-pod.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: tensorboard
spec:
  containers:
  - name: tensorboard
    image: tensorflow/tensorflow:latest
    command: ["tensorboard", "--logdir=/logs", "--host=0.0.0.0", "--port=6006"]
    ports:
    - containerPort: 6006
    volumeMounts:
    - name: storage
      mountPath: /logs
      subPath: logs
  volumes:
  - name: storage
    persistentVolumeClaim:
      claimName: v2v-diffuser-kuntal
```

### 6.2 Access TensorBoard

```bash
# Deploy TensorBoard
kubectl apply -f kub_files/tensorboard-pod.yaml

# Port forward to access locally
kubectl port-forward pod/tensorboard 6006:6006
```

Open http://localhost:6006 in your browser.

## Step 7: Troubleshooting

### Common Issues

#### Pod Not Starting
```bash
# Check pod status and events
kubectl describe pod <pod-name>

# Check logs
kubectl logs <pod-name>
```

#### GPU Not Available
```bash
# Check if GPU nodes exist
kubectl get nodes -L nvidia.com/gpu.product

# Remove or adjust nodeSelector in YAML files
```

#### Out of Memory
- Reduce `batch_size` in `config/cloud_train_config.yaml`
- Increase memory limits in pod spec
- Enable gradient checkpointing in config

#### Storage Issues
```bash
# Check PVC status
kubectl get pvc v2v-diffuser-kuntal

# Check if PVC is bound
kubectl describe pvc v2v-diffuser-kuntal
```

#### Image Pull Errors
- Ensure your Docker image is pushed to registry
- Check image name in YAML files
- Verify registry credentials if using private registry

## Step 8: Clean Up

### Stop Training

```bash
# Delete job
kubectl delete job v2v-diffusion-training

# Or delete pod
kubectl delete pod v2v-diffusion-training-pod
```

### Delete All Resources (Caution: This deletes data!)

```bash
# Delete all resources
kubectl delete -f kub_files/training-job.yaml
kubectl delete -f kub_files/training-pod.yaml
kubectl delete -f kub_files/interactive-pod.yaml

# Delete PVC (WARNING: This deletes all training data!)
# kubectl delete pvc v2v-diffuser-kuntal
```

## Configuration Tips

### Adjusting for Different GPU Types

| GPU Type | Batch Size | Memory Request |
|----------|------------|----------------|
| V100 (16GB) | 1-2 | 32Gi |
| V100 (32GB) | 2-4 | 48Gi |
| A100 (40GB) | 4-6 | 48Gi |
| A100 (80GB) | 8-12 | 64Gi |

### Training Time Estimates

- **With Pretrained Weights**: ~24-48 hours on A100 (50 epochs)
- **From Scratch**: ~7-10 days on A100 (100 epochs)

### Optimizing Training Speed

1. **Enable Mixed Precision**: Already enabled in `cloud_train_config.yaml`
2. **Increase Batch Size**: Adjust based on GPU memory
3. **Use Gradient Accumulation**: If batch size is limited
4. **Enable Gradient Checkpointing**: Trade compute for memory

## Next Steps

1. Monitor training progress via logs
2. Evaluate checkpoints periodically
3. Adjust hyperparameters if needed
4. Run inference on validation data
5. Deploy best model for inference

## Support

For issues specific to:
- **Nautilus/NRP**: Check https://nrp.ai/documentation/
- **Training Code**: Check project README and issues
- **Kubernetes**: Refer to kubectl documentation

## Summary of Files Created

```
kub_files/
├── persistent_storage.yaml      # PVC for data storage
├── interactive-pod.yaml          # Interactive pod for testing
├── training-job.yaml             # Training job (recommended)
├── training-pod.yaml             # Training pod (alternative)
└── DEPLOYMENT_GUIDE.md           # This guide

Dockerfile                        # Docker image definition
.dockerignore                     # Docker build optimization
```
