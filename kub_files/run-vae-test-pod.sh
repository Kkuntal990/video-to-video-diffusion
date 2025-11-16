#!/bin/bash
# Temporary pod to run VAE reconstruction tests on Kubernetes
# This uses the same image as training jobs with access to pretrained checkpoints

kubectl run -it --rm vae-test-pod \
  --image=kuntalkokate/llm_agent_v2v_train:latest \
  --restart=Never \
  --overrides='
  {
    "spec": {
      "containers": [
        {
          "name": "vae-test",
          "image": "kuntalkokate/llm_agent_v2v_train:latest",
          "stdin": true,
          "tty": true,
          "command": ["/bin/bash"],
          "workingDir": "/workspace",
          "volumeMounts": [
            {
              "name": "storage",
              "mountPath": "/workspace/storage_a100"
            },
            {
              "name": "dshm",
              "mountPath": "/dev/shm"
            }
          ],
          "resources": {
            "requests": {
              "memory": "16Gi",
              "cpu": "4",
              "nvidia.com/gpu": "1"
            },
            "limits": {
              "memory": "16Gi",
              "cpu": "4",
              "nvidia.com/gpu": "1"
            }
          },
          "env": [
            {
              "name": "PYTHONUNBUFFERED",
              "value": "1"
            },
            {
              "name": "TORCH_HOME",
              "value": "/workspace/storage_a100/.cache/torch"
            },
            {
              "name": "HF_HOME",
              "value": "/workspace/storage_a100/.cache/huggingface"
            }
          ]
        }
      ],
      "volumes": [
        {
          "name": "storage",
          "persistentVolumeClaim": {
            "claimName": "v2v-diffuser-kuntal-a100-shared"
          }
        },
        {
          "name": "dshm",
          "emptyDir": {
            "medium": "Memory",
            "sizeLimit": "8Gi"
          }
        }
      ],
      "affinity": {
        "nodeAffinity": {
          "preferredDuringSchedulingIgnoredDuringExecution": [
            {
              "weight": 100,
              "preference": {
                "matchExpressions": [
                  {
                    "key": "nvidia.com/gpu.product",
                    "operator": "In",
                    "values": ["Tesla-V100-SXM2-32GB", "Tesla-V100-SXM2-16GB"]
                  }
                ]
              }
            }
          ]
        }
      }
    }
  }'
