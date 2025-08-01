#!/usr/bin/env python3
"""
Script to check Elasticsearch status and diagnose issues
"""

import os
import sys
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError

def check_elasticsearch_status():
    """Check Elasticsearch connection and index status"""
    
    # Configuration
    host = os.getenv("ELASTICSEARCH_HOST", "elasticsearch")
    port = os.getenv("ELASTICSEARCH_PORT", "9200")
    index_name = "aic2025"
    
    print("🔍 Checking Elasticsearch Status")
    print("=" * 50)
    
    # Create client
    try:
        es = Elasticsearch(
            hosts=[f"http://{host}:{port}"],
            verify_certs=False,
            request_timeout=30,
            max_retries=3,
            retry_on_timeout=True
        )
    except Exception as e:
        print(f"❌ Failed to create Elasticsearch client: {e}")
        return False
    
    # Check connection
    print(f"📍 Connecting to Elasticsearch at {host}:{port}...")
    try:
        if es.ping():
            print("✅ Elasticsearch is reachable")
        else:
            print("❌ Elasticsearch is not responding")
            return False
    except ConnectionError as e:
        print(f"❌ Connection failed: {e}")
        print("💡 Check if Elasticsearch is running:")
        print("   - Run: docker ps")
        print("   - If not running: docker-compose up -d")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    # Get cluster info
    try:
        cluster_info = es.info()
        print(f"📊 Cluster name: {cluster_info['cluster_name']}")
        print(f"📊 Elasticsearch version: {cluster_info['version']['number']}")
    except Exception as e:
        print(f"⚠️  Could not get cluster info: {e}")
    
    # Check index status
    print(f"\n📁 Checking index: {index_name}")
    try:
        if es.indices.exists(index=index_name):
            # Get index stats
            stats = es.indices.stats(index=index_name)
            index_stats = stats['indices'][index_name]
            
            print(f"✅ Index exists")
            print(f"   📊 Documents: {index_stats['total']['docs']['count']:,}")
            print(f"   💾 Size: {index_stats['total']['store']['size_in_bytes'] / (1024*1024):.2f} MB")
            print(f"   🗂️  Shards: {index_stats['total']['shards']['total']}")
            
            # Check if index is healthy
            health = es.cluster.health(index=index_name)
            status = health['status']
            if status == 'green':
                print(f"   🟢 Status: {status} (healthy)")
            elif status == 'yellow':
                print(f"   🟡 Status: {status} (warning)")
            else:
                print(f"   🔴 Status: {status} (unhealthy)")
                
        else:
            print(f"❌ Index does not exist")
            
    except Exception as e:
        print(f"⚠️  Could not check index status: {e}")
    
    # Check cluster health
    print(f"\n🏥 Cluster Health:")
    try:
        health = es.cluster.health()
        status = health['status']
        if status == 'green':
            print(f"   🟢 Status: {status}")
        elif status == 'yellow':
            print(f"   🟡 Status: {status}")
        else:
            print(f"   🔴 Status: {status}")
            
        print(f"   📊 Number of nodes: {health['number_of_nodes']}")
        print(f"   📊 Active shards: {health['active_shards']}")
        print(f"   📊 Relocating shards: {health['relocating_shards']}")
        print(f"   📊 Initializing shards: {health['initializing_shards']}")
        print(f"   📊 Unassigned shards: {health['unassigned_shards']}")
        
    except Exception as e:
        print(f"⚠️  Could not get cluster health: {e}")
    
    # Check disk space
    print(f"\n💾 Disk Usage:")
    try:
        stats = es.nodes.stats()
        for node_id, node_stats in stats['nodes'].items():
            fs_stats = node_stats.get('fs', {})
            if 'total' in fs_stats:
                total_bytes = fs_stats['total']['total_in_bytes']
                available_bytes = fs_stats['total']['available_in_bytes']
                used_bytes = total_bytes - available_bytes
                used_percent = (used_bytes / total_bytes) * 100
                
                print(f"   📊 Total: {total_bytes / (1024**3):.2f} GB")
                print(f"   📊 Used: {used_bytes / (1024**3):.2f} GB ({used_percent:.1f}%)")
                print(f"   📊 Available: {available_bytes / (1024**3):.2f} GB")
                
                if used_percent > 90:
                    print(f"   ⚠️  Warning: Disk usage is high ({used_percent:.1f}%)")
                break
    except Exception as e:
        print(f"⚠️  Could not get disk usage: {e}")
    
    print("\n" + "=" * 50)
    return True

def main():
    """Main function"""
    success = check_elasticsearch_status()
    
    if success:
        print("✅ Elasticsearch status check completed")
    else:
        print("❌ Elasticsearch status check failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 