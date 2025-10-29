"""
Ray Train version of distributed LLM fine-tuning for wilderness survival Q&A.
This script uses Ray Train with TorchTrainer to replace manual PyTorch DDP setup.
"""

import torch
import os
import json
import tempfile
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import SFTTrainer, SFTConfig

# Ray Train imports
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig

# Optional: Ray Tune for hyperparameter optimization
try:
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    RAY_TUNE_AVAILABLE = True
except ImportError:
    RAY_TUNE_AVAILABLE = False
    print("Ray Tune not available - skipping hyperparameter tuning features")

# Optional: Aim tracking (may not be in MODH image)
try:
    from aim.hugging_face import AimCallback
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False
    print("Aim not available - skipping experiment tracking")


# Wilderness survival expert system prompt (same as original)
WILDERNESS_EXPERT_SYSTEM_PROMPT = """You are a wilderness survival and practical skills expert. Your mission is to provide comprehensive, detailed guidance on essential survival and practical skills. Give thorough, step-by-step instructions with explanations of why each step matters.

Your expertise covers:
- Wilderness Survival Basics: Rule of 3s (3 minutes without air, 3 hours without shelter in harsh conditions, 3 days without water, 3 weeks without food), emergency signaling techniques, essential knots, identifying poisonous plants and safe alternatives
- Basic First Aid: Treatment for cuts, burns, sprains, shock, and emergency care procedures
- Simple Car Maintenance: Checking fluids (oil, coolant, brake, transmission), tire inspection and pressure, lights and electrical systems
- Basic Cooking Techniques: Food safety, preparation methods, cooking over open fires, food preservation
- Common Measurement Conversions: Imperial to metric, cooking measurements, distance and weight conversions
- Essential Knots: Bowline, clove hitch, trucker's hitch, figure-eight, sheet bend, and their practical applications

Always provide detailed explanations, safety warnings when relevant, and multiple approaches when possible. Your responses should be comprehensive enough to help someone learn and apply these skills safely and effectively. Aim for thorough, educational responses rather than brief answers."""


def format_prompt(example):
    """Format Q&A data for TRL SFTTrainer"""
    question = example['full-question']
    answer = example['answer']
    
    messages = [
        {"role": "system", "content": WILDERNESS_EXPERT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
    
    return {"messages": messages}


# Custom callbacks (same as original)
class TrainingHistoryCallback(TrainerCallback):
    def __init__(self):
        self.history = []
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs:
            self.history.append(logs)


class PerplexityCallback(TrainerCallback):
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Calculate and log perplexity from loss values"""
        if logs:
            import math
            if 'loss' in logs:
                logs['perplexity'] = math.exp(logs['loss'])
            if 'eval_loss' in logs:
                logs['eval_perplexity'] = math.exp(logs['eval_loss'])


def train_func(config):
    """
    Main training function that runs on each Ray worker.
    Ray Train handles all distributed setup automatically.
    
    Args:
        config: Dictionary with training configuration parameters (unused - for compatibility)
    """
    # Get Ray Train distributed context (replaces manual setup)
    world_size = train.get_context().get_world_size()
    rank = train.get_context().get_world_rank()
    local_rank = train.get_context().get_local_rank()
    
    print(f"Ray Train Worker - Rank {rank}/{world_size}, Local Rank {local_rank}")
    
    # Configuration from environment variables (EXACTLY like original training.py)
    model_name = "HuggingFaceTB/SmolLM3-3B"
    dataset_path = os.getenv("DATASET_PATH", "data/qa_dataset.parquet")
    epochs = int(os.getenv("EPOCHS", "4"))
    batch_size = int(os.getenv("BATCH_SIZE", "8"))
    learning_rate = float(os.getenv("LEARNING_RATE", "5e-5"))
    data_dir = os.getenv("DATA_DIR", "/tmp/data")
    output_dir = os.getenv("OUTPUT_DIR", "/tmp/models")
    aim_repo = os.getenv("AIM_REPO", "/aim")
    
    # Create output directories (Ray handles coordination)
    os.makedirs(f"{output_dir}/best_model", exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    
    print(f"Configuration:")
    print(f"- Model: {model_name}")
    print(f"- Dataset: {dataset_path}")
    print(f"- Epochs: {epochs}")
    print(f"- Batch size: {batch_size}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Output dir: {output_dir}")
    print(f"- World size: {world_size}")
    
    # Check CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available - training on CPU (slow!)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model and Tokenizer
    print("Loading model and tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/tmp/model_cache")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("Tokenizer loaded successfully")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        cache_dir="/tmp/model_cache",
        attn_implementation="flash_attention_2",
    )
    
    print(f"Model loaded successfully")
    
    # Monitor VRAM usage after model loading
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"VRAM after model loading: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    # Freeze all layers except the last ones for memory efficiency
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze last 2 layers and output layer
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
            for param in layers[-2:].parameters():
                param.requires_grad = True
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
            for param in layers[-2:].parameters():
                param.requires_grad = True
        elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
            layers = model.gpt_neox.layers
            for param in layers[-2:].parameters():
                param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = True
        
        # Unfreeze output embedding layer
        if hasattr(model, 'embed_out'):
            for param in model.embed_out.parameters():
                param.requires_grad = True
        elif hasattr(model, 'lm_head'):
            for param in model.lm_head.parameters():
                param.requires_grad = True
    
    except Exception as e:
        print(f"Warning during layer unfreezing: {e}")
        for param in model.parameters():
            param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Load and Prepare Dataset
    print(f"Loading Q&A dataset from: {dataset_path}")
    
    dataset = load_dataset("parquet", data_files=dataset_path)
    full_dataset = dataset["train"]
    
    print(f"Total dataset size: {len(full_dataset):,} Q&A pairs")
    
    split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    
    # Apply formatting for TRL SFTTrainer
    print("Processing datasets for TRL SFTTrainer...")
    processed_train = train_dataset.map(format_prompt)
    processed_val = val_dataset.map(format_prompt)
    
    # Set up callbacks
    history_callback = TrainingHistoryCallback()
    perplexity_callback = PerplexityCallback()
    
    callbacks = [history_callback, perplexity_callback]
    
    # Add Aim callback if available
    if AIM_AVAILABLE:
        aim_callback = AimCallback(
            repo=aim_repo,  # From environment variable
            experiment='smollm3-wilderness-ray-finetuning'
        )
        callbacks.append(aim_callback)
    
    # Configure training arguments
    # Ray Train handles distributed setup automatically via Accelerate
    training_args = SFTConfig(
        output_dir=f"{output_dir}/checkpoints",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        max_grad_norm=1.0,
        num_train_epochs=epochs,
        logging_steps=2,
        logging_first_step=True,
        save_steps=500,
        bf16=True,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataloader_drop_last=True,
        max_seq_length=1024,
        packing=False,
        # Ray Train will configure distributed backend automatically
    )
    
    # Create SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        eval_dataset=processed_val,
        callbacks=callbacks,
    )
    
    # Start Fine-Tuning
    print("Starting fine-tuning on Q&A dataset with SmolLM3-3B...")
    print(f"Ray Train distributed training with {world_size} workers")
    
    training_result = trainer.train()
    print("Fine-tuning complete!")
    
    # Save model and artifacts (Ray handles coordination)
    print("Saving model and training artifacts...")
    
    # Create temporary directory for checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save model and tokenizer
        trainer.save_model(tmpdir)
        tokenizer.save_pretrained(tmpdir)
        
        # Save training history
        history_path = os.path.join(tmpdir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(history_callback.history, f, indent=2)
        
        # Save training summary
        training_summary = {
            "model_name": model_name,
            "dataset_path": dataset_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "training_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
            "final_train_loss": training_result.training_loss if hasattr(training_result, 'training_loss') else None,
            "world_size": world_size,
            "distributed_training": world_size > 1,
            "framework": "Ray Train"
        }
        
        summary_path = os.path.join(tmpdir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(training_summary, f, indent=2)
        
        # Report metrics and checkpoint to Ray Train
        metrics = {
            "final_loss": training_result.training_loss if hasattr(training_result, 'training_loss') else None,
            "trainable_params": trainable_params,
            "total_params": total_params,
        }
        
        # Get eval loss if available
        if history_callback.history:
            last_log = history_callback.history[-1]
            if 'eval_loss' in last_log:
                metrics['eval_loss'] = last_log['eval_loss']
        
        # Report to Ray Train with checkpoint
        from ray.train import Checkpoint
        checkpoint = Checkpoint.from_directory(tmpdir)
        train.report(metrics, checkpoint=checkpoint)
    
    print(f"\nRay Train distributed fine-tuning completed successfully!")
    print(f"Trained on {world_size} GPUs")
    print(f"Final training loss: {metrics.get('final_loss', 'N/A')}")
    
    return training_result


def main():
    """
    Main entry point for Ray Train distributed training.
    Sets up TorchTrainer and launches distributed training.
    
    Configuration is read from environment variables (same as original training.py).
    """
    # Configuration from environment variables (EXACTLY like original training.py)
    model_name = "HuggingFaceTB/SmolLM3-3B"
    dataset_path = os.getenv("DATASET_PATH", "data/qa_dataset.parquet")
    epochs = int(os.getenv("EPOCHS", "4"))
    batch_size = int(os.getenv("BATCH_SIZE", "8"))
    learning_rate = float(os.getenv("LEARNING_RATE", "5e-5"))
    data_dir = os.getenv("DATA_DIR", "/tmp/data")
    output_dir = os.getenv("OUTPUT_DIR", "/tmp/models")
    aim_repo = os.getenv("AIM_REPO", "/aim")
    
    # Scaling configuration for distributed training
    num_workers = int(os.getenv("NUM_WORKERS", "4"))
    
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=True,
        resources_per_worker={"CPU": 6, "GPU": 1},
    )
    
    # Run configuration with checkpointing
    run_config = RunConfig(
        name="smollm3-wilderness-ray-finetuning",
        storage_path=output_dir,
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute="eval_loss",
            checkpoint_score_order="min",
        ),
    )
    
    print("=" * 80)
    print("Ray Train Distributed LLM Fine-tuning")
    print("=" * 80)
    print(f"Configuration from environment variables:")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Data dir: {data_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Num workers: {num_workers}")
    print("=" * 80)
    
    # Empty config dict (train_func reads from env vars directly)
    config = {}
    
    # Create TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=config,  # Empty - train_func uses env vars
        scaling_config=scaling_config,
        run_config=run_config,
    )
    
    # Start distributed training
    print("\nStarting Ray Train distributed training...")
    result = trainer.fit()
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    
    # Get best checkpoint
    if result.best_checkpoints:
        best_checkpoint = result.best_checkpoints[0][0]
        print(f"Best checkpoint: {best_checkpoint}")
        
        # Copy best checkpoint to final output directory
        final_output = os.path.join(config["output_dir"], "best_model")
        print(f"Copying best model to: {final_output}")
        best_checkpoint.to_directory(final_output)
        print("Model saved successfully!")
    
    print("\nTraining metrics:")
    print(f"  Final metrics: {result.metrics}")
    print("=" * 80)
    
    return result


if __name__ == "__main__":
    # Check if running inside Ray Train worker
    # If so, this is called by TorchTrainer, run train_func directly
    # If not, set up TorchTrainer and launch distributed training
    try:
        # Try to get Ray Train context - if this works, we're inside a worker
        context = train.get_context()
        print("Running inside Ray Train worker - this shouldn't happen when called directly")
        print("Use 'ray job submit' or CodeFlare SDK to launch this script")
    except RuntimeError:
        # Not inside Ray Train worker - run main setup
        main()

