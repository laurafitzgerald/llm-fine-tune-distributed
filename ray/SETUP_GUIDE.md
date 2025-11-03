# Ray Distributed Training Setup Guide

## Overview

This guide provides step-by-step instructions for setting up and running distributed LLM fine-tuning with Ray Train and CodeFlare SDK on RHOAI.

---

## Prerequisites

- **RHOAI Installation** - Red Hat OpenShift AI installed and configured
- **Data Science Project** - A Data Science Project created in RHOAI
- **Workbench** - A workbench created within your Data Science Project
  - **Image:** Select `Jupyter | Data Science | CPU | Python 3.12` from the notebook image dropdown
  - **Container size:** Small or Medium is sufficient for running scripts
- **Namespace** - Note your Data Science Project namespace (you'll need this for all commands)
  - To find your namespace: Check the Data Science Project name in RHOAI dashboard
  - Format: Usually matches your project name (e.g., `my-project`)

**Note:** All steps in this guide are performed from within your Jupyter workbench terminal.

---

## Steps

### Step 1: Clone the Repository in Jupyter

Clone this repository into your Jupyter workbench.

**Using Terminal:**

```bash
# Navigate to home directory
cd ~

# Clone repository from laurafitzgerald remote
git clone https://github.com/laurafitzgerald/llm-fine-tune-distributed.git

# Navigate to repository
cd llm-fine-tune-distributed

# Checkout ray branch
git checkout ray

# Navigate to scripts directory
cd ray/scripts

# Verify you're on the ray branch
git branch
# Should show: * ray
```

---

### Step 2: Verify OpenShift Login

From the Jupyter workbench terminal, check if you are logged in to OpenShift.

**Command:**
```bash
oc whoami
```

**Expected Output (if logged in):**
```
your-username
```

**If you see this error:**
```
error: You must be logged in to the server (Unauthorized)
```
OR output like
```
system:service-account
```

**Then you must login to OpenShift:**

1. In the OpenShift web console, click your username (top right)
2. Click **"Copy login command"**
3. Click **"Display Token"**
4. Copy the `oc login` command
5. Paste and run it in your workbench terminal

**Example:**
```bash
oc login --token=sha256~xxxxx --server=https://api.your-cluster.com:6443
```

**Verify login succeeded:**
```bash
oc whoami
# Should show your username
```

---

### Step 3: Verify CodeFlare SDK Version

Check that you have CodeFlare SDK version 0.32.0 installed.

**Command:**
```bash
python3 -c "import codeflare_sdk; print(f'CodeFlare SDK version: {codeflare_sdk.__version__}')"
```

**Expected Output:**
```
CodeFlare SDK version: 0.32.0
```

**If you see a different version or an error, update CodeFlare SDK:**

```bash
pip install --upgrade codeflare-sdk==0.32.0
```

**Verify the update:**
```bash
python3 -c "import codeflare_sdk; print(f'CodeFlare SDK version: {codeflare_sdk.__version__}')"
# Should now show: CodeFlare SDK version: 0.32.0
```

---

### Step 4: Verify GPU Node Requirements

Before creating the Ray cluster, ensure you have enough GPU nodes available.

**This Ray cluster configuration requires:**
- **Minimum 4 GPU nodes** (1 head + 3 workers)
- **Each node needs:** 6+ vCPUs, 20+ GB RAM, 1 GPU

**Check if you have enough GPU nodes:**
```bash
oc get nodes -l nvidia.com/gpu.present=true
# Should show at least 4 GPU nodes
```

**Verify GPU availability:**
```bash
oc get nodes -l nvidia.com/gpu.present=true -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}'
# Should show 4 nodes with 1 GPU each
```

---

#### Adding GPU Nodes via OCM Console

**If you don't have enough GPU nodes**, add them via the **OpenShift Cluster Manager (OCM) console**:

1. **Navigate to:** OCM Console → Your Cluster → **Machine Pools** tab
2. **Click:** "Add machine pool"
3. **Configure machine pool:**

**Recommended AWS Instance Types:**

| Instance Type | vCPUs | RAM | GPU | GPU VRAM | Best For | Cost/hr |
|--------------|-------|-----|-----|----------|----------|---------|
| **g5.2xlarge** ⭐ | 8 | 32GB | 1x A10G | 24GB | **Recommended** | ~$1.21 |
| **g5.4xlarge** | 16 | 64GB | 1x A10G | 24GB | Large models | ~$1.62 |
| **g6.2xlarge** | 8 | 32GB | 1x L40S | 48GB | Best perf (L40S) | ~$1.50 |
| **g4dn.2xlarge** | 8 | 32GB | 1x T4 | 16GB | Budget option | ~$0.75 |

**Machine Pool Settings:**
- **Instance type:** `g5.2xlarge` (recommended)
- **Node count:** 4 (minimum for this Ray cluster)
- **Min replicas:** 4
- **Max replicas:** 4
- **Labels:** `node-role.kubernetes.io/gpu: ""` (optional)

**After adding the machine pool, wait for nodes to be ready (5-10 minutes):**
```bash
# Watch GPU nodes becoming ready
watch oc get nodes -l nvidia.com/gpu.present=true

# Once ready, verify GPU availability
oc get nodes -l nvidia.com/gpu.present=true -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}'
# Should show 4 nodes with 1 GPU each
```

---

### Step 5: Create Ray Cluster

Now create the Ray cluster for distributed training.

**First, export your namespace as an environment variable:**

By exporting `NAMESPACE`, you can reuse it in all subsequent commands without modification!

```bash
# Replace with your actual Data Science Project namespace
export NAMESPACE=<YOUR_NAMESPACE>

# Verify it's set
echo "Using namespace: $NAMESPACE"
```

**Example:**
```bash
export NAMESPACE=my-project
echo "Using namespace: $NAMESPACE"
```

**Now create the cluster:**
```bash
python3 1_setup_cluster.py \
  --namespace $NAMESPACE \
  --head-cpus 6 \
  --head-memory 20 \
  --worker-cpus 6 \
  --worker-memory 20 \
  --head-gpus 1 \
  --worker-gpus 1 \
  --num-workers 3
```

**Expected output:**
```
================================================================================
RAY CLUSTER SETUP
================================================================================
Cluster Configuration:
  Name: smollm3-ray-cluster
  Namespace: your-namespace
  Head: 6 CPU, 20GB RAM, 1 GPU
  Workers: 3x (6 CPU, 20GB RAM, 1 GPU)
  Total: 24 CPU, 80GB RAM, 4 GPU
  
Deploying cluster...
✓ Cluster deployment initiated
  Waiting for cluster to be ready (timeout: 600s)...

✓ CLUSTER IS READY!

================================================================================
CLUSTER STATUS
================================================================================
Status: ready

CodeFlare Cluster Details:
  Name: smollm3-ray-cluster
  Namespace: your-namespace
  Workers: 3
  Dashboard: http://smollm3-ray-cluster-head-svc.your-namespace.svc.cluster.local:8265
  ...

✅ CLUSTER SETUP COMPLETE
```

**Important:** Note the **Dashboard** link in the **CodeFlare Cluster Details** output. You can use this to monitor your training job after submission.

---

### Step 6: Submit Training Job

Now submit the training job to the Ray cluster.

**Important:** Job names must be unique in OpenShift. Use a unique job name for each submission.

**Command:**
```bash
# Use timestamp for uniqueness
python3 2_submit_job.py \
  --namespace $NAMESPACE \
  --job-name training-$(date +%Y%m%d-%H%M%S) \
  --batch-size 6 \
  --epochs 4 \
  --monitor
```

**This will:**
- Submit a RayJob to the cluster with the specified unique name
- Configure training with batch size 6 and 4 epochs
- Monitor the job progress until completion

**Expected output:**
```
================================================================================
SUBMITTING RAYJOB CR
================================================================================
Job name: smollm3-ray-cluster-training-job
...
✅ RAYJOB SUBMITTED SUCCESSFULLY!
...
MONITORING JOB
================================================================================
[HH:MM:SS] Job Status: RUNNING
...
```

### Step 7: Cleanup Ray Cluster

When training is complete, clean up the Ray cluster to free resources.

**Command:**
```bash
python3 3_cleanup_cluster.py --namespace $NAMESPACE
```

**This will:**
- Prompt for confirmation before deleting
- Delete the RayCluster resource
- Delete all Ray pods and services
- Verify cleanup completed

**Expected output:**
```
================================================================================
RAY CLUSTER CLEANUP
================================================================================
Cluster: smollm3-ray-cluster
Namespace: your-namespace
...

✓ Found cluster
  Status: ready

⚠️  WARNING: This will delete:
  - RayCluster: smollm3-ray-cluster
  - All Ray pods and services
  - Any running jobs
================================================================================

Proceed with deletion? (yes/no): yes

Deleting RayCluster...
✓ Cluster deletion initiated
  Resources are being cleaned up...

✓ Cluster deleted (status check failed as expected)
...

✅ CLEANUP COMPLETE
```

**To skip confirmation prompt (force delete):**
```bash
python3 3_cleanup_cluster.py --namespace $NAMESPACE --force
```

**To verify cleanup manually:**
```bash
# Check if RayCluster is deleted
oc get raycluster -n $NAMESPACE

# Check if Ray pods are deleted
oc get pods -l ray.io/cluster=smollm3-ray-cluster -n $NAMESPACE

# Both should show "No resources found" or empty
```

---


