#!/bin/bash
# Monitor training pod memory and GPU usage in real-time

POD_NAME=${1:-v2v-diffusion-training-pod}

echo "==================================="
echo "Training Pod Monitor"
echo "==================================="
echo "Pod: $POD_NAME"
echo "Press Ctrl+C to stop monitoring"
echo "==================================="
echo ""

# Check if pod exists
if ! kubectl get pod $POD_NAME >/dev/null 2>&1; then
    echo "Error: Pod '$POD_NAME' not found!"
    echo ""
    echo "Usage: $0 [pod-name]"
    echo "Example: $0 v2v-diffusion-training-pod"
    exit 1
fi

# Monitor loop - update every 10 seconds
watch -n 10 "
echo '=================================================='
echo 'TIMESTAMP: '\$(date)
echo '=================================================='
echo ''
echo '--- POD METRICS (kubectl top) ---'
kubectl top pod $POD_NAME 2>&1 || echo 'Pod metrics not available yet'
echo ''
echo '--- SYSTEM MEMORY (free -h) ---'
kubectl exec $POD_NAME -- free -h 2>&1 || echo 'Cannot access pod shell'
echo ''
echo '--- GPU USAGE (nvidia-smi) ---'
kubectl exec $POD_NAME -- nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv 2>&1 || echo 'Cannot access GPU info'
echo ''
echo '=================================================='
echo 'Monitoring every 10 seconds... (Ctrl+C to stop)'
echo '=================================================='
"
