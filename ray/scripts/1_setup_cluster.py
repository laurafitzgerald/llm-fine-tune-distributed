#!/usr/bin/env python3
"""
Step 1: Setup and Create Ray Cluster
Creates a Ray cluster on OpenShift AI using CodeFlare SDK.
"""

import os
import sys
import argparse
from datetime import datetime

# CodeFlare SDK imports
try:
    from codeflare_sdk import Cluster, ClusterConfiguration, TokenAuthentication
except ImportError:
    print("ERROR: CodeFlare SDK not installed")
    print("Install with: pip install codeflare-sdk")
    sys.exit(1)


def create_cluster(
    cluster_name="smollm3-ray-cluster",
    namespace="lyric-professor",
    num_workers=3,
    head_cpus=2,
    head_memory=8,
    head_gpus=1,
    worker_cpus=2,
    worker_memory=8,
    worker_gpus=1,
    timeout=600,
):
    """
    Create and deploy Ray cluster.
    
    Args:
        cluster_name: Name of the Ray cluster
        namespace: OpenShift namespace/project
        num_workers: Number of worker nodes (head is additional)
        head_cpus: CPUs for head node
        head_memory: Memory in GB for head node
        head_gpus: GPUs for head node
        worker_cpus: CPUs per worker
        worker_memory: Memory in GB per worker
        worker_gpus: GPUs per worker
        timeout: Timeout in seconds to wait for cluster ready
    
    Returns:
        Cluster object
    """
    print("=" * 80)
    print("RAY CLUSTER SETUP")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    print(f"Cluster Configuration:")
    print(f"  Name: {cluster_name}")
    print(f"  Namespace: {namespace}")
    print(f"  Head: {head_cpus} CPU, {head_memory}GB RAM, {head_gpus} GPU")
    print(f"  Workers: {num_workers}x ({worker_cpus} CPU, {worker_memory}GB RAM, {worker_gpus} GPU)")
    print(f"  Total: {head_cpus + num_workers*worker_cpus} CPU, "
          f"{head_memory + num_workers*worker_memory}GB RAM, "
          f"{head_gpus + num_workers*worker_gpus} GPU")
    print("=" * 80)
    
    # Create ClusterConfiguration
    print("\nCreating ClusterConfiguration...")
    cluster_config = ClusterConfiguration(
        name=cluster_name,
        namespace=namespace,
        num_workers=num_workers,
        # CPU and memory configuration (requests and limits)
        head_cpu_requests=head_cpus,
        head_memory_requests=head_memory,
        worker_cpu_requests=worker_cpus,
        worker_memory_requests=worker_memory,
        head_cpu_limits=head_cpus,
        head_memory_limits=head_memory,
        worker_cpu_limits=worker_cpus,
        worker_memory_limits=worker_memory,
        write_to_file=False,
        # GPU configuration
        head_extended_resource_requests={"nvidia.com/gpu": head_gpus} if head_gpus > 0 else {},
        worker_extended_resource_requests={"nvidia.com/gpu": worker_gpus} if worker_gpus > 0 else {},
    )
    
    print("✓ ClusterConfiguration created")
    
    # Create and deploy cluster
    print("\nDeploying cluster...")
    print("  Creating Kubernetes resources:")
    print("    - RayCluster custom resource")
    print(f"    - Head pod (1x)")
    print(f"    - Worker pods ({num_workers}x)")
    print("    - Services (head service, dashboard)")
    print("")
    
    try:
        cluster = Cluster(cluster_config)
        cluster.apply()
        print("✓ Cluster deployment initiated")
    except Exception as e:
        print(f"\n✗ FAILED to submit cluster creation to Kubernetes!")
        print(f"   Error: {e}")
        print("\nPossible causes:")
        print("  - kubectl not configured or no access to namespace")
        print("  - Insufficient permissions to create RayCluster resources")
        print(f"  - Namespace '{namespace}' does not exist")
        print("  - KubeRay operator not installed")
        print("\nDebug commands:")
        print(f"  kubectl auth can-i create rayclusters -n {namespace}")
        print(f"  kubectl get namespace {namespace}")
        print(f"  kubectl api-resources | grep raycluster")
        print("\n✗ EXITING DUE TO CLUSTER SUBMISSION FAILURE")
        sys.exit(1)
    
    print(f"  Waiting for cluster to be ready (timeout: {timeout}s)...")
    
    try:
        cluster.wait_ready(timeout=timeout)
        print("\n✓ CLUSTER IS READY!")
    except Exception as e:
        print(f"\n✗ CLUSTER FAILED TO BECOME READY!")
        print(f"   Error: {e}")
        print("\nChecking cluster status...")
        try:
            status = cluster.status()
            print(f"  Status: {status}")
            print("\nCluster details:")
            print(cluster.details())
        except Exception as detail_error:
            print(f"  Could not retrieve details: {detail_error}")
        
        print("\nManual debugging:")
        print(f"  kubectl get raycluster {cluster_name} -n {namespace}")
        print(f"  kubectl describe raycluster {cluster_name} -n {namespace}")
        print(f"  kubectl get pods -l ray.io/cluster={cluster_name} -n {namespace}")
        print(f"  kubectl logs -l ray.io/cluster={cluster_name} -n {namespace} --tail=50")
        print("\n✗ EXITING DUE TO CLUSTER CREATION FAILURE")
        sys.exit(1)
    
    # Display cluster status
    print("\n" + "=" * 80)
    print("CLUSTER STATUS")
    print("=" * 80)
    
    try:
        status = cluster.status()
        print(f"Status: {status}")
        print("\nDetails:")
        print(cluster.details())
    except Exception as e:
        print(f"✗ Error getting cluster status: {e}")
        print("  Cluster may have been created but status is unavailable")
        print("\nManual check:")
        print(f"  kubectl get raycluster {cluster_name} -n {namespace}")
        sys.exit(1)
    
    # Display Ray Dashboard access
    print("\n" + "=" * 80)
    print("RAY DASHBOARD ACCESS")
    print("=" * 80)
    dashboard_url = cluster.cluster_dashboard_uri()
    print(f"Dashboard URL: {dashboard_url}")
    print("=" * 80)
    
    return cluster


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Setup Ray cluster for distributed training")
    
    # Cluster configuration
    parser.add_argument("--cluster-name", default="smollm3-ray-cluster", help="Cluster name")
    parser.add_argument("--namespace", default="lyric-professor", help="OpenShift namespace")
    parser.add_argument("--num-workers", type=int, default=3, help="Number of worker nodes")
    
    # Resource configuration
    parser.add_argument("--head-cpus", type=int, default=2, help="Head CPUs")
    parser.add_argument("--head-memory", type=int, default=8, help="Head memory (GB)")
    parser.add_argument("--head-gpus", type=int, default=1, help="Head GPUs")
    parser.add_argument("--worker-cpus", type=int, default=2, help="Worker CPUs")
    parser.add_argument("--worker-memory", type=int, default=8, help="Worker memory (GB)")
    parser.add_argument("--worker-gpus", type=int, default=1, help="Worker GPUs")
    
    # Timeout
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds")
    
    args = parser.parse_args()
    
    # Create cluster
    cluster = create_cluster(
        cluster_name=args.cluster_name,
        namespace=args.namespace,
        num_workers=args.num_workers,
        head_cpus=args.head_cpus,
        head_memory=args.head_memory,
        head_gpus=args.head_gpus,
        worker_cpus=args.worker_cpus,
        worker_memory=args.worker_memory,
        worker_gpus=args.worker_gpus,
        timeout=args.timeout,
    )
    
    print("\n" + "=" * 80)
    print("✅ CLUSTER SETUP COMPLETE")
    print("=" * 80)
    print(f"Cluster '{args.cluster_name}' is ready in namespace '{args.namespace}'")
    print("")
    print("Next steps:")
    print("  1. Submit training job:")
    print("     python 2_submit_job.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

