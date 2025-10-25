#!/bin/bash
# Script to verify checkpoints are saved to persistent storage in Kubernetes

set -e

POD_NAME="v2v-diffusion-training-pod"

echo "=========================================="
echo "Checking Persistent Storage Setup"
echo "=========================================="
echo ""

# Check if PVC exists
echo "1. Checking PersistentVolumeClaim..."
kubectl get pvc v2v-diffuser-kuntal -o wide 2>/dev/null || {
    echo "❌ PVC 'v2v-diffuser-kuntal' not found!"
    echo "Create it first: kubectl apply -f kub_files/persistent_storage.yaml"
    exit 1
}
echo "✓ PVC exists"
echo ""

# Check if pod exists
echo "2. Checking if pod is running..."
kubectl get pod $POD_NAME 2>/dev/null || {
    echo "❌ Pod '$POD_NAME' not found!"
    echo "You can check for other pods with: kubectl get pods | grep v2v"
    exit 1
}

POD_STATUS=$(kubectl get pod $POD_NAME -o jsonpath='{.status.phase}')
echo "✓ Pod status: $POD_STATUS"
echo ""

# Check if storage is mounted
echo "3. Checking if persistent storage is mounted..."
kubectl exec $POD_NAME -- df -h /workspace/storage 2>/dev/null || {
    echo "❌ Storage not mounted!"
    exit 1
}
echo "✓ Persistent storage is mounted"
echo ""

# Check checkpoint directory
echo "4. Checking checkpoint directory..."
kubectl exec $POD_NAME -- ls -lh /workspace/storage/checkpoints/ 2>/dev/null || {
    echo "⚠️  Checkpoint directory doesn't exist yet (will be created during training)"
}
echo ""

# Check logs directory
echo "5. Checking logs directory..."
kubectl exec $POD_NAME -- ls -lh /workspace/storage/logs/ 2>/dev/null || {
    echo "⚠️  Logs directory doesn't exist yet (will be created during training)"
}
echo ""

# Show storage usage
echo "6. Storage usage:"
kubectl exec $POD_NAME -- du -sh /workspace/storage/* 2>/dev/null || {
    echo "⚠️  No data in storage yet"
}
echo ""

echo "=========================================="
echo "✓ Verification Complete!"
echo "=========================================="
echo ""
echo "Checkpoints will be saved to: /workspace/storage/checkpoints/"
echo "This path is backed by PVC: v2v-diffuser-kuntal (20Gi)"
echo ""
echo "To monitor checkpoints during training:"
echo "  kubectl exec $POD_NAME -- watch -n 5 ls -lh /workspace/storage/checkpoints/"
echo ""
echo "To copy checkpoints to local machine:"
echo "  kubectl cp $POD_NAME:/workspace/storage/checkpoints ./local_checkpoints"
