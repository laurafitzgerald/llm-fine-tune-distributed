# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Primary Goal for this Project:**

I am learning Red Hat OpenShift AI for distributed model training with Kubeflow. This project implements **wilderness survival and practical skills Q&A fine-tuning** using **distributed training across multiple GPUs** as a demonstration of scalable AI capabilities. The goal is to transform a generic language model into a wilderness survival expert using distributed computing for faster, more efficient training.

**Project Evolution Context:**
This project evolved from a single-node training setup to a distributed training architecture. We successfully implemented PyTorchJob distributed training with OpenShift AI monitoring using Kubeflow. We then added a **Ray/KubeRay implementation** using Ray Train with CodeFlare SDK, which provides significant advantages in development workflow, monitoring, and fault tolerance.

**Current Phase: Dual Implementation Approach**
The project now supports TWO distributed training approaches:

1. **PyTorchJob** Using Kubeflow Training Operator with manual PyTorch DDP setup
2. **Ray/KubeRay**  Using Ray Train with CodeFlare SDK for simplified deployment

Both implement distributed TRL (Transformers Reinforcement Learning) framework for advanced chat model fine-tuning across multiple NVIDIA L40S GPUs. Ray is recommended for development due to faster iteration (no Docker builds needed with MODH images).

**Target Demo Scenario:**
- **Before Fine-tuning**: Generic model provides basic responses to survival questions
- **After Fine-tuning**: Specialized wilderness survival expert provides comprehensive, detailed guidance with step-by-step instructions, safety warnings, and educational explanations
- **Performance Improvement**: 4x faster training with distributed setup (1 Master + 3 Workers)

**Dataset**: HuggingFace "cahlen/offline-practical-skills-qa-synthetic" (2,845 practical skills Q&A pairs covering wilderness survival, car maintenance, home repairs, computer troubleshooting, and other practical skills).

**Model from Hugging Face**: "HuggingFaceTB/SmolLM3-3B" (upgraded from Pythia-410m for better performance)

**Training Technique**: Distributed BF16 fine-tuning using TRL SFTTrainer with PyTorch Distributed Data Parallel (DDP), last layer unfrozen, and comprehensive wilderness survival expert system prompt.

## Distributed Training Architecture

### PyTorchJob Implementation

**PyTorch Distributed Setup:**
- **Master Node**: 1 replica - coordinates training and handles model saving
- **Worker Nodes**: 3 replicas - participate in distributed training
- **Communication Backend**: NCCL for GPU-to-GPU communication
- **Total World Size**: 4 (1 Master + 3 Workers)
- **Deployment**: YAML-based with kubectl
- **Files**: `training.py`, `deploy/pytorchjob.yaml`, `deploy/deploy-script.sh`

### Ray/KubeRay Implementation ⚡ NEW

**Ray Train Distributed Setup:**
- **Head Node**: 1 node - Ray cluster coordinator + training participant
- **Worker Nodes**: 3 nodes - Ray workers for distributed training
- **Communication**: Ray Train with automatic distributed setup (handles NCCL internally)
- **Deployment**: Programmatic with CodeFlare SDK (Python) via ray-distributed-training.ipynb
- **Files**: `ray_training.py`

**Ray Advantages:**
- **No Docker Builds for Dev**: CodeFlare MODH image (`quay.io/modh/ray:2.23.0-py39-cu121`) includes Ray + PyTorch + CUDA
- **Runtime Environment**: Upload code and install packages dynamically via `runtime_env`
- **Ray Dashboard**: Rich monitoring UI with metrics, task timelines, resource utilization
- **Automatic Distributed Setup**: No manual NCCL/DDP configuration needed

**Distributed Training Benefits (Both Implementations):**
- **4x Faster Training**: Parallel processing across 4 GPUs
- **Linear Scaling**: Each GPU processes different data batches simultaneously
- **Memory Efficiency**: Model parameters shared, gradients synchronized
- **Fault Tolerance**: Training continues if individual workers fail

## NVIDIA L40S GPU Specifications (Per Node)

The project uses NVIDIA L40S GPUs optimized for AI/ML workloads in distributed configuration:

**Core Architecture:**
- **GPU Architecture**: NVIDIA Ada Lovelace architecture
- **CUDA Cores**: 18,176 NVIDIA Ada Lovelace Architecture-Based CUDA® Cores
- **RT Cores**: 142 NVIDIA Third-Generation RT Cores
- **Tensor Cores**: 568 NVIDIA Fourth-Generation Tensor Cores

**Memory:**
- **GPU Memory**: 48GB GDDR6 with ECC per GPU
- **Memory Bandwidth**: 864GB/s per GPU
- **Total Distributed VRAM**: 192GB (48GB × 4 GPUs)
- **Interconnect**: PCIe Gen4 x16 (64GB/s bidirectional)

**Performance Specifications (Per GPU):**
- **FP32 Performance**: 91.6 TFLOPS
- **RT Core Performance**: 212 TFLOPS
- **TF32 Tensor**: 183/366* TFLOPS
- **BFLOAT16 Tensor**: 362.05/733* TFLOPS
- **FP16 Tensor**: 362.05/733* TFLOPS
- **FP8 Tensor**: 733/1,466* TFLOPS
- **INT8 Tensor**: 733/1,466* TOPS
- **INT4 Tensor**: 733/1,466* TOPS

**Distributed Performance (4 GPUs):**
- **Total BFLOAT16 Performance**: ~1,448 TFLOPS (362.05 × 4)
- **Aggregate Memory Bandwidth**: 3,456 GB/s (864 × 4)
- **Effective Training Speedup**: ~4x compared to single GPU

*Performance numbers with asterisk indicate performance with sparsity optimization

**Key Technical Decisions:**
- **Distributed TRL Framework**: Using SFTTrainer with SFTConfig optimized for distributed chat model fine-tuning
- **Distributed Options**: PyTorch DDP (manual) OR Ray Train (automatic)
- **CodeFlare MODH Images**: Pre-built Ray images for fast development (Ray implementation)
- **SmolLM3-3B**: Larger, more capable model architecture suitable for distributed training
- **No RAG (Retrieval-Augmented Generation)**: Pure fine-tuning approach for reliable demo
- **BF16 Training**: Optimal precision for L40S Ada Lovelace architecture across all nodes
- **NCCL Backend**: High-performance GPU communication (automatically configured by Ray Train)
- **Synchronized Batch Processing**: Coordinated training across all GPUs

**Red Hat OpenShift AI Distributed Cluster Details:**
- **Total Nodes**: 4 (1 Master + 3 Workers)
- **Per Node Resources**:
    - RAM: 20 GB (optimized for distributed training memory requirements)
    - CPU: 6 vCPU
    - GPU: 1× NVIDIA L40S (48 GB VRAM)
- **Total Cluster Resources**:
    - RAM: 80 GB total (20 GB × 4 nodes)
    - CPU: 24 vCPU total (6 × 4 nodes)
    - GPU: 4× NVIDIA L40S (192 GB total VRAM)
- **Network**: High-speed interconnect for NCCL communication

**Namespace/Project**: lyric-professor (consistent across all phases)

**Dataset Integration:**
- **Source**: Downloaded from HuggingFace `cahlen/offline-practical-skills-qa-synthetic`
- **Format**: Converted from JSONL to Parquet for efficient loading (77.7% size reduction)
- **Schema**: Two columns - "full-question" (format: "For [topic], [question]") and "answer"
- **Containerization**: Dataset embedded directly in Docker container (no runtime download required)
- **Processing**: TRL-compatible message format with comprehensive wilderness survival system prompt
- **Distribution**: Same dataset loaded on all nodes (no data sharding needed for this size)

**Storage Configuration:**
- **trained-models-pvc**: Shared model outputs and checkpoints (mounted on all nodes)
- **workspace-pvc**: Shared working directory and temporary files (mounted on all nodes)
- **Data**: Embedded in container (no separate PVC needed)

### Environment Variables

**PyTorchJob Training (`training.py`):**
- `EPOCHS`: Number of training epochs (default: 4)
- `BATCH_SIZE`: Training batch size per GPU (default: 8)
- `LEARNING_RATE`: Learning rate (default: 5e-5, scaled by world size in PyTorchJob)
- `WORLD_SIZE`: Total number of processes (set automatically by PyTorchJob)
- `RANK`: Process rank (set automatically by PyTorchJob)
- `LOCAL_RANK`: Local GPU rank within node (set automatically)
- `MASTER_ADDR`: Master node address (set by PyTorchJob)
- `MASTER_PORT`: Master communication port (default: 23456)
- `NCCL_DEBUG`: NCCL debugging level (default: INFO)
- `DATA_DIR`: Directory for dataset (default: /shared/data)
- `OUTPUT_DIR`: Directory for model outputs (default: /shared/models)

**Ray Training (`ray_training.py`):**
- `NUM_WORKERS`: Number of Ray workers for training (default: 4)
- `OUTPUT_DIR`: Directory for model outputs (default: /tmp/models)
- `DATASET_PATH`: Path to parquet dataset (default: data/qa_dataset.parquet)
- `AIM_REPO`: Aim tracking repository path (default: /aim)

**Note**: Ray Train handles distributed context automatically - no WORLD_SIZE, RANK, MASTER_ADDR needed!

### Distributed Training Optimizations

**Batch Size Scaling:**
- **Per-Device Batch Size**: 8 (reduced from single-node 12 for stability)
- **Gradient Accumulation**: 2 steps (reduced from single-node 5)
- **Effective Batch Size**: 8 × 2 GPUs × 2 accumulation = 32
- **Training Efficiency**: Optimized for memory usage and convergence

**Learning Rate Scaling:**
- **Base Learning Rate**: 5e-5
- **Distributed Scaling**: 5e-5 × 4 = 2e-4 (linear scaling with world size)
- **Rationale**: Compensates for larger effective batch size

**Communication Optimization:**
- **DDP Backend**: NCCL for optimal GPU-to-GPU communication
- **Bucket Cap**: 25MB for gradient bucketing efficiency
- **Find Unused Parameters**: False (optimized for model architecture)

### Wilderness Survival Expert System Prompt:

The model is trained with a comprehensive system prompt that establishes it as a wilderness survival and practical skills expert covering:

**Core Expertise Areas:**
- **Wilderness Survival Basics**: Rule of 3s, emergency signaling, essential knots, plant identification
- **Basic First Aid**: Treatment for cuts, burns, sprains, shock, emergency procedures
- **Simple Car Maintenance**: Fluid checks, tire inspection, electrical systems
- **Basic Cooking Techniques**: Food safety, preparation, cooking over fires, preservation
- **Common Measurement Conversions**: Imperial to metric, cooking measurements, distances
- **Essential Knots**: Bowline, clove hitch, trucker's hitch, figure-eight, sheet bend

**Response Guidelines:**
- Provide thorough, step-by-step instructions with explanations
- Include safety warnings when relevant
- Offer multiple approaches when possible
- Educational responses rather than brief answers
- Comprehensive enough for practical application

### Dataset Details:

**Source Information:**
- **Origin**: HuggingFace dataset `cahlen/offline-practical-skills-qa-synthetic`
- **License**: Synthetic dataset for educational/training purposes
- **Size**: 2,845 Q&A pairs (714KB original JSONL → 159KB Parquet)
- **Topics**: Wilderness survival, car maintenance, home repairs, computer troubleshooting, practical life skills

**Data Processing Pipeline:**
1. **Download**: Original JSONL format from HuggingFace
2. **Transform**: Concatenate topic + question into "full-question" field
3. **Convert**: JSONL → Parquet for efficient ML loading
4. **Embed**: Include dataset directly in Docker container
5. **Load**: Use HuggingFace datasets library with local parquet file
6. **Runtime Format**: Convert to TRL chat template format during training via `format_prompt()` function

**Sample Prompt Structure:**
```
messages: [
  {"role": "system", "content": "You are a wilderness survival expert..."},
  {"role": "user", "content": "For Simple Car Maintenance Checks, What is the recommended tire pressure for my car?"},
  {"role": "assistant", "content": "Check your car's owner's manual or the tire information placard on the driver's side doorjamb for the recommended tire pressure."}
]
```

### Output Artifacts (saved to `OUTPUT_DIR` by rank 0 only):
- `best_model/`: Best performing model checkpoint with tokenizer
- `training_history.json`: Epoch-by-epoch metrics and logs
- `training_summary.json`: Training configuration and results (includes distributed info)
- `demo_outputs/`: Sample Q&A outputs for demo scenarios
- `checkpoints/`: Training checkpoints for recovery

### Inference Scripts:
- `ask_tuned_model.py`: Query the fine-tuned wilderness survival expert
- `ask_original_model.py`: Query the original SmolLM3-3B for comparison
- Both scripts use the same comprehensive system prompt for fair comparison

**OpenShift AI Integration:**
- PyTorchJob annotated for distributed metrics collection in OpenShift AI console
- Resource usage monitoring via OpenShift AI → Distributed workloads → Project metrics
- GPU utilization and memory usage tracking across all nodes
- Updated for distributed wilderness survival training focus

**Training Framework:**

**PyTorchJob Implementation:**
- **Distributed TRL SFTTrainer**: Advanced supervised fine-tuning for chat models with manual DDP
- **SFTConfig**: Optimized training arguments with explicit distributed parameters
- **Manual Setup**: setup_distributed() function configures NCCL backend
- **Accelerate Integration**: Handles DDP wrapping and gradient synchronization
- **Rank 0 Saving**: Only master node saves model artifacts

**Ray Train Implementation:**
- **Ray TorchTrainer**: Wraps TRL SFTTrainer for distributed execution
- **Automatic Setup**: Ray Train handles all distributed configuration
- **train_func**: Training logic wrapped in function for distribution
- **Ray Dashboard**: Real-time monitoring and debugging

**Common to Both:**
- **TRL SFTTrainer**: Advanced supervised fine-tuning for chat models
- **SFTConfig**: Optimized training arguments for conversational AI
- **Chat Template**: Proper ChatML format handling across all nodes
- **System Prompt Integration**: Consistent training and inference prompts
- **BF16 Precision**: Optimal for L40S Ada Lovelace architecture
- **Gradient Synchronization**: Automatic (PyTorch DDP or Ray Train)

**Training Metrics:**
- Training/validation loss curves (from rank 0)
- Learning rate scheduling
- Gradient norms and optimization stability
- Sample throughput and training speed (distributed)
- Model parameter efficiency (13.62% trainable parameters)
- GPU utilization across all nodes

## Performance Optimizations for Distributed NVIDIA L40S

**Distributed TRL Framework Benefits:**
- **Optimized Distributed Chat Training**: Purpose-built for multi-GPU conversational AI fine-tuning
- **Memory Efficiency**: Better memory management across distributed nodes
- **Template Handling**: Automatic chat template processing on all GPUs
- **Advanced Features**: Support for distributed training techniques

**L40S-Specific Distributed Performance Benefits:**
- **8x Memory Capacity**: 192GB total enables larger models and longer sequences
- **Enhanced Tensor Performance**: 1,448+ TFLOPS for distributed BF16 operations
- **Ada Lovelace Architecture**: 4th-gen Tensor Cores with improved AI/ML efficiency across all GPUs
- **High Memory Bandwidth**: 3,456 GB/s aggregate for faster distributed data throughput

**Current Distributed Optimization Settings:**
- **Batch Size**: 8 per device with gradient accumulation of 2 (effective batch size: 64)
- **Sequence Length**: 1024 tokens for comprehensive responses
- **BF16 Training**: Leveraging L40S Ada Lovelace optimization across all GPUs
- **Gradient Checkpointing**: Enabled for memory efficiency
- **Flash Attention 2**: Optimized attention computation on all nodes
- **NCCL Communication**: Optimized for L40S interconnect performance

**Model Architecture:**
- **Base Model**: SmolLM3-3B (3.075B total parameters)
- **Trainable Parameters**: 418.9M parameters (13.62% of total)
- **Training Strategy**: Last 2 transformer layers + language modeling head
- **Memory Efficient**: Partial layer unfreezing for optimal distributed VRAM usage
- **Gradient Synchronization**: Efficient parameter updates across all GPUs

**Distributed Training Performance Metrics:**
- **Training Speed**: ~4x faster than single-node training
- **Memory Utilization**: Optimal distribution across 192GB total VRAM
- **Communication Overhead**: Minimized through NCCL optimization
- **Convergence**: Improved due to larger effective batch size and learning rate scaling







