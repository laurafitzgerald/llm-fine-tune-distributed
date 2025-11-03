#!/usr/bin/env python3
"""
Step 2: Submit Training Job and Monitor
Submits distributed training job to existing Ray cluster and monitors progress.
"""

import os
import sys
import time
import argparse
from datetime import datetime

# CodeFlare SDK imports
try:
    from codeflare_sdk import Cluster, ClusterConfiguration, RayJob
except ImportError:
    print("ERROR: CodeFlare SDK not installed")
    print("Install with: pip install codeflare-sdk")
    sys.exit(1)


def submit_and_monitor_job(
    cluster_name="smollm3-ray-cluster",
    namespace="default",
    job_name=None,
    epochs=4,
    batch_size=8,
    learning_rate=5e-5,
    output_dir="/tmp/models",
    dataset_path="data/qa_dataset.parquet",
    aws_access_key_id=None,
    aws_secret_access_key=None,
    aws_region=None,
    monitor=True,
    follow_logs=False,
):
    """
    Submit training job to Ray cluster and optionally monitor progress.
    
    Args:
        cluster_name: Name of the existing Ray cluster
        namespace: OpenShift namespace
        epochs: Number of training epochs
        batch_size: Batch size per GPU
        learning_rate: Learning rate
        output_dir: Output directory for model
        monitor: Monitor job status
        follow_logs: Display logs during monitoring
    
    Returns:
        submission_id: Job submission ID
    """
    print("=" * 80)
    print("RAY JOB SUBMISSION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    # Connect to existing cluster
    print(f"Connecting to cluster '{cluster_name}' in namespace '{namespace}'...")
    
    try:
        # Recreate cluster configuration to connect to existing cluster
        cluster_config = ClusterConfiguration(
            name=cluster_name,
            namespace=namespace,
        )
        cluster = Cluster(cluster_config)
        
        # Check cluster status
        status = cluster.status()
        print(f"✓ Connected to cluster")
        print(f"  Cluster status: {status}")
        
        if status != "ready":
            print(f"\n⚠️  WARNING: Cluster status is '{status}', expected 'ready'")
            print("  Job submission may fail if cluster is not ready")
            
    except Exception as e:
        print(f"✗ Failed to connect to cluster: {e}")
        print(f"\nVerify cluster exists:")
        print(f"  kubectl get raycluster {cluster_name} -n {namespace}")
        raise
    
    # Define runtime environment
    # OPTION 1: Use command-line arguments (avoids env_vars string issues!)
    print("\nPreparing runtime environment...")
    
    # Build entrypoint with command-line arguments
    # Since working_dir is ./scripts, ray_training.py is at the root of uploaded dir
    # Note: CPUS_PER_WORKER set via env var to match cluster worker_cpus
    entrypoint = (
        f"python ray_training.py "
        f"--epochs {epochs} "
        f"--batch-size {batch_size} "
        f"--learning-rate {learning_rate} "
        f"--output-dir {output_dir} "
        f"--dataset-path {dataset_path}"
    )
    
    # Runtime env without env_vars (much simpler!)
    # Exclude data directory to avoid making secret too large
    # List packages directly (instead of -r requirements.txt which causes path issues)
    # Note: flash-attn removed - it requires compilation and torch to be pre-installed
    #       Use a custom image if you need flash-attn
    runtime_env = {
        "working_dir": ".",  # Upload current directory (we're in scripts/)
        "pip": [
            "accelerate>=1.6.0",
            "transformers>=4.46.0",
            "trl>=0.8.0",
            "datasets",
            "aim",
            "s3fs",  # For S3 dataset loading
            # "flash-attn>=2.0.0",  # Removed - requires torch pre-installed and compilation
        ],
        "excludes": ["data/", "*.parquet", "*.jsonl"],  # Exclude data files
        "env_vars": {
            # Set CPUS_PER_WORKER to match cluster worker_cpus (important!)
            # This tells TorchTrainer how many CPUs to request per worker
            "CPUS_PER_WORKER": "4",  # Must match cluster worker_cpus
            # AWS credentials for S3 access (if provided)
            **({
                "AWS_ACCESS_KEY_ID": aws_access_key_id,
                "AWS_SECRET_ACCESS_KEY": aws_secret_access_key,
            } if aws_access_key_id and aws_secret_access_key else {}),
            **({
                "AWS_DEFAULT_REGION": aws_region,
            } if aws_region else {}),
        }
    }
    
    print(f"  Working dir: {runtime_env['working_dir']} (current directory)")
    print(f"  Packages: {len(runtime_env['pip'])} packages (listed directly)")
    for pkg in runtime_env['pip']:
        print(f"    - {pkg}")
    print(f"  Excludes: {', '.join(runtime_env['excludes'])}")
    print(f"  Entrypoint: {entrypoint}")
    print("✓ Using command-line arguments (no env_vars conversion issues!)")
    print("⚠️  Note: Data files excluded from upload (must be in container image or PVC)")
    
    # OPTION 2: If you prefer env_vars, use hardcoded strings
    # runtime_env = {
    #     "working_dir": "./scripts",
    #     "pip": ["-r requirements.txt"],
    #     "env_vars": {
    #         "EPOCHS": "4",  # Hardcoded string literals
    #         "BATCH_SIZE": "8",
    #         "LEARNING_RATE": "5e-5",  # Hardcoded string literal
    #         "OUTPUT_DIR": "/tmp/models",
    #     }
    # }
    # entrypoint = "python ray_training.py"
    
    # Use provided job_name or generate default
    if job_name is None:
        job_name = f"{cluster_name}-training-job"
    
    print("\n" + "=" * 80)
    print("CREATING RAYJOB CR")
    print("=" * 80)
    print(f"Job name: {job_name}")
    print(f"Entrypoint: {entrypoint}")
    print(f"Cluster: {cluster_name}")
    print(f"Namespace: {namespace}")
    print("")
    
    try:
        # Create RayJob object using CodeFlare SDK
        ray_job = RayJob(
            job_name=job_name,
            cluster_name=cluster_name,
            namespace=namespace,
            entrypoint=entrypoint,
            runtime_env=runtime_env,
        )
        
        print("✓ RayJob object created")
        print("")
        
        # Submit RayJob CR to Kubernetes
        print("Submitting RayJob CR to Kubernetes...")
        submission_id = ray_job.submit()
        
        print("")
        print("✅ RAYJOB CR SUBMITTED SUCCESSFULLY!")
        print(f"  Job Name: {job_name}")
        print(f"  Submission ID: {submission_id}")
        print("")
        print("A RayJob Kubernetes resource has been created.")
        print("")
        print("Check RayJob status:")
        print(f"  kubectl get rayjob {job_name} -n {namespace}")
        print(f"  kubectl describe rayjob {job_name} -n {namespace}")
        print("=" * 80)
        
        # Store ray_job for monitoring
        job_client = ray_job
        
    except Exception as e:
        print(f"\n✗ FAILED TO CREATE/SUBMIT RAYJOB CR!")
        print(f"   Error: {e}")
        print("\nTroubleshooting:")
        print(f"  1. Check cluster exists:")
        print(f"     kubectl get raycluster {cluster_name} -n {namespace}")
        print(f"  2. Check cluster is ready:")
        print(f"     kubectl get raycluster {cluster_name} -n {namespace} -o jsonpath='{{.status.state}}'")
        print(f"  3. Verify namespace access:")
        print(f"     kubectl auth can-i create rayjobs -n {namespace}")
        print(f"  4. Check CodeFlare SDK version:")
        print(f"     python3 -c 'import codeflare_sdk; print(codeflare_sdk.__version__)'")
        print("\n✗ EXITING DUE TO RAYJOB SUBMISSION FAILURE")
        sys.exit(1)
    
    # Monitor job if requested
    if monitor:
        print("\n" + "=" * 80)
        print("MONITORING JOB")
        print("=" * 80)
        print("Polling job status every 30 seconds... (Ctrl+C to stop)")
        print("")
        
        try:
            while True:
                try:
                    # Get status from RayJob object
                    status = job_client.status()
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    print(f"[{timestamp}] RayJob Status: {status}")
                    
                    # Show logs if requested
                    if follow_logs:
                        try:
                            logs = job_client.logs()
                            if logs:
                                print(f"  Latest logs:\n{logs[-500:]}")  # Last 500 chars
                        except:
                            pass
                    
                    # Check for terminal states
                    status_str = str(status).upper()
                    if any(term in status_str for term in ["SUCCEEDED", "FAILED", "STOPPED"]):
                        print(f"\n✓ RayJob finished with status: {status}")
                        break
                    
                except Exception as status_error:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Error checking status: {status_error}")
                
                time.sleep(30)
                
        except KeyboardInterrupt:
            print("\n⏸️  Monitoring stopped by user")
            try:
                current_status = job_client.status()
                print(f"   Current RayJob status: {current_status}")
            except:
                print("   Could not retrieve current status")
        except Exception as e:
            print(f"\n✗ Monitoring error: {e}")
            print("  Job may still be running. Check manually:")
            print(f"  kubectl get rayjob {job_name} -n {namespace}")
            sys.exit(1)
    
    # Display final status
    print("\n" + "=" * 80)
    print("FINAL RAYJOB STATUS")
    print("=" * 80)
    
    try:
        final_status = job_client.status()
        print(f"RayJob Status: {final_status}")
        print(f"Job Name: {job_name}")
        print("")
        
        status_str = str(final_status).upper()
        if "SUCCEEDED" in status_str:
            print("✅ Training completed successfully!")
            print(f"\nModel saved to: {output_dir}/best_model/")
            print(f"  {output_dir}/training_history.json")
            print(f"  {output_dir}/training_summary.json")
            print("\nTo retrieve model:")
            print(f"  kubectl exec -it <head-pod> -n {namespace} -- bash")
            print(f"  cd {output_dir} && ls -la")
        elif "FAILED" in status_str:
            print("❌ Training failed!")
            print("\nCheck logs:")
            print(f"  kubectl logs -l ray.io/node-type=head -n {namespace} --tail=100")
            print(f"  kubectl describe rayjob {job_name} -n {namespace}")
        elif "RUNNING" in status_str:
            print("⏳ RayJob is still running")
            print("  Use --monitor flag to track progress")
            print(f"  Or check: kubectl get rayjob {job_name} -n {namespace}")
        else:
            print(f"Status: {final_status}")
            
    except Exception as e:
        print(f"Error getting final status: {e}")
        print("\nCheck RayJob manually:")
        print(f"  kubectl get rayjob {job_name} -n {namespace}")
        print(f"  kubectl describe rayjob {job_name} -n {namespace}")
    
    print("=" * 80)
    
    return submission_id


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Submit training job to Ray cluster")
    
    # Cluster identification
    parser.add_argument("--cluster-name", default="smollm3-ray-cluster", help="Cluster name")
    parser.add_argument("--namespace", default="lyric-professor", help="OpenShift namespace")
    parser.add_argument("--job-name", default=None, help="Job name (default: <cluster-name>-training-job)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--output-dir", default="/tmp/models", help="Output directory")
    parser.add_argument("--dataset-path", default="data/qa_dataset.parquet", 
                       help="Path to dataset (local, S3, or PVC mount)")
    
    # AWS/S3 configuration (optional - for S3 datasets)
    parser.add_argument("--aws-access-key-id", default=os.getenv("AWS_ACCESS_KEY_ID"),
                       help="AWS Access Key ID (for S3 datasets)")
    parser.add_argument("--aws-secret-access-key", default=os.getenv("AWS_SECRET_ACCESS_KEY"),
                       help="AWS Secret Access Key (for S3 datasets)")
    parser.add_argument("--aws-region", default=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
                       help="AWS region (for S3 datasets)")
    
    # Monitoring options
    parser.add_argument("--monitor", action="store_true", help="Monitor job until completion")
    parser.add_argument("--follow-logs", action="store_true", help="Show logs during monitoring")
    
    args = parser.parse_args()
    
    # Set job name (use provided or generate from cluster name)
    if args.job_name is None:
        args.job_name = f"{args.cluster_name}-training-job"
    
    # Submit and monitor job
    submission_id = submit_and_monitor_job(
        cluster_name=args.cluster_name,
        namespace=args.namespace,
        job_name=args.job_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        aws_region=args.aws_region,
        monitor=args.monitor,
        follow_logs=args.follow_logs,
    )
    
    print("\n" + "=" * 80)
    print("✅ RAYJOB SUBMISSION COMPLETE")
    print("=" * 80)
    print(f"Submission ID: {submission_id}")
    print(f"Job Name: {args.job_name}")
    print("")
    print("RayJob Kubernetes resource:")
    print(f"  kubectl get rayjob {args.job_name} -n {args.namespace}")
    print(f"  kubectl describe rayjob {args.job_name} -n {args.namespace}")
    print("")
    print("Next steps:")
    if not args.monitor:
        print(f"  Monitor job: python3 2_submit_job.py --job-name {args.job_name} --monitor")
    print(f"  View logs: kubectl logs -l ray.io/node-type=head -n {args.namespace} --tail=100")
    print(f"  When done: python3 3_cleanup_cluster.py")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

