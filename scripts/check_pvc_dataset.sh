#!/bin/bash
# Check PVC contents to find dataset location

POD_NAME="pvc-checker-$(date +%s)"

echo "Creating temporary pod to check PVC contents..."

kubectl run $POD_NAME \
  --image=alpine:latest \
  --restart=Never \
  --overrides='
{
  "spec": {
    "containers": [{
      "name": "checker",
      "image": "alpine:latest",
      "command": ["sh", "-c", "sleep 3600"],
      "volumeMounts": [{
        "name": "storage",
        "mountPath": "/workspace/storage_a100"
      }]
    }],
    "volumes": [{
      "name": "storage",
      "persistentVolumeClaim": {
        "claimName": "v2v-diffuser-kuntal-a100"
      }
    }]
  }
}'

echo "Waiting for pod to be ready..."
kubectl wait --for=condition=Ready pod/$POD_NAME --timeout=60s

echo ""
echo "=== PVC Root Directory ==="
kubectl exec $POD_NAME -- ls -lah /workspace/storage_a100/

echo ""
echo "=== Looking for dataset directories ==="
kubectl exec $POD_NAME -- find /workspace/storage_a100/ -maxdepth 3 -type d -name "*APE*" -o -name "*dataset*" -o -name "*.zip" 2>/dev/null | head -20

echo ""
echo "=== Storage usage ==="
kubectl exec $POD_NAME -- df -h /workspace/storage_a100/

echo ""
echo "Cleaning up..."
kubectl delete pod $POD_NAME

echo ""
echo "Done! Check the output above to find your dataset location."
