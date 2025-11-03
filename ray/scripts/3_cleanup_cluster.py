#!/usr/bin/env python3
"""
Step 3: Cleanup Ray Cluster
Deletes Ray cluster and all associated resources.
"""

import os
import sys
import time
import argparse
from datetime import datetime

# CodeFlare SDK imports
try:
    from codeflare_sdk import Cluster, ClusterConfiguration
except ImportError:
    print("ERROR: CodeFlare SDK not installed")
    print("Install with: pip install codeflare-sdk")
    sys.exit(1)


def cleanup_cluster(
    cluster_name="smollm3-ray-cluster",
    namespace="lyric-professor",
    force=False,
):
    """
    Delete Ray cluster and verify cleanup.
    
    Args:
        cluster_name: Name of the Ray cluster to delete
        namespace: OpenShift namespace
        force: Skip confirmation prompt
    
    Returns:
        True if cleanup successful
    """
    print("=" * 80)
    print("RAY CLUSTER CLEANUP")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    print(f"Cluster: {cluster_name}")
    print(f"Namespace: {namespace}")
    print("=" * 80)
    
    # Check if cluster exists first (using kubectl directly to avoid CodeFlare messages)
    print("\nChecking if cluster exists...")
    import subprocess
    result = subprocess.run(
        ["oc", "get", "raycluster", cluster_name, "-n", namespace],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"✓ Cluster '{cluster_name}' does not exist - nothing to clean up")
        return True
    
    print(f"✓ Found cluster '{cluster_name}'")
    
    # Connect to cluster for managed cleanup
    try:
        cluster_config = ClusterConfiguration(
            name=cluster_name,
            namespace=namespace,
        )
        cluster = Cluster(cluster_config)
        
        # Get cluster status (suppress "No resources" messages)
        try:
            status = cluster.status()
            print(f"  Status: {status}")
        except Exception:
            # Cluster exists in k8s but CodeFlare can't get status - that's ok
            print(f"  Status: Found in Kubernetes")
            
    except Exception as e:
        print(f"⚠️  Could not connect via CodeFlare: {e}")
        print("  Will use oc commands for cleanup")
        cluster = None
    
    # Confirmation prompt (unless force)
    if not force:
        print("\n" + "=" * 80)
        print("⚠️  WARNING: This will delete:")
        print(f"  - RayCluster: {cluster_name}")
        print("  - All Ray pods and services")
        print("  - Any running jobs")
        print("=" * 80)
        
        response = input("\nProceed with deletion? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Cleanup cancelled.")
            return False
    
    # Delete cluster
    print("\n" + "=" * 80)
    print("DELETING CLUSTER")
    print("=" * 80)
    
    deletion_successful = False
    
    # Try CodeFlare SDK cleanup first
    if cluster:
        try:
            print("Initiating cluster deletion via CodeFlare SDK...")
            cluster.down()
            print("✓ Cluster deletion initiated")
            deletion_successful = True
        except Exception as e:
            print(f"⚠️  CodeFlare deletion failed: {e}")
            print("  Falling back to oc commands...")
    
    # Fallback to oc commands if needed
    if not deletion_successful:
        print("Deleting cluster via oc command...")
        result = subprocess.run(
            ["oc", "delete", "raycluster", cluster_name, "-n", namespace],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✓ Cluster deletion initiated")
            deletion_successful = True
        else:
            print(f"✗ Deletion failed: {result.stderr}")
            raise RuntimeError(f"Could not delete cluster {cluster_name}")
    
    # Wait for deletion
    print("\nWaiting for resources to be deleted...")
    time.sleep(10)
    print("✓ Deletion complete")
    
    # Verify cleanup
    print("\n" + "=" * 80)
    print("VERIFYING CLEANUP")
    print("=" * 80)
    
    # Check via oc command (avoid CodeFlare "No resources" message)
    result = subprocess.run(
        ["oc", "get", "raycluster", cluster_name, "-n", namespace],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("✓ Cluster deleted successfully")
    else:
        print(f"⚠️  Cluster still exists")
        print("  Wait a few more seconds and check again")
        return False
    
    # Verify with oc
    print("\nVerifying via oc...")
    
    # Check RayCluster
    result = subprocess.run(
        ["oc", "get", "raycluster", cluster_name, "-n", namespace],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("  ✓ No RayCluster found")
    else:
        print(f"  ⚠️  RayCluster still exists:\n{result.stdout}")
    
    # Check pods
    result = subprocess.run(
        ["oc", "get", "pods", "-l", f"ray.io/cluster={cluster_name}", "-n", namespace],
        capture_output=True,
        text=True
    )
    if "No resources found" in result.stderr or not result.stdout.strip():
        print("  ✓ No Ray pods found")
    else:
        print(f"  ⚠️  Ray pods still exist:\n{result.stdout}")
    
    print("\n" + "=" * 80)
    print("✅ CLEANUP COMPLETE")
    print("=" * 80)
    
    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Cleanup Ray cluster")
    
    # Cluster identification
    parser.add_argument("--cluster-name", default="smollm3-ray-cluster", help="Cluster name")
    parser.add_argument("--namespace", default="lyric-professor", help="OpenShift namespace")
    
    # Options
    parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompt")
    
    args = parser.parse_args()
    
    # Cleanup cluster
    success = cleanup_cluster(
        cluster_name=args.cluster_name,
        namespace=args.namespace,
        force=args.force,
    )
    
    if success:
        print("\n✅ All resources have been cleaned up successfully!")
        return 0
    else:
        print("\n⚠️  Cleanup may not be complete. Check manually.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


