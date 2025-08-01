#!/usr/bin/env python3
"""
Script to manually delete Elasticsearch index
"""

import os
import sys
import time
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError

def delete_index():
    """Delete the Elasticsearch index with retry logic"""
    
    # Configuration
    host = os.getenv("ELASTICSEARCH_HOST", "elasticsearch")
    port = os.getenv("ELASTICSEARCH_PORT", "9200")
    index_name = "aic2025"
    
    print("🗑️  Deleting Elasticsearch Index")
    print("=" * 50)
    
    # Create client
    try:
        es = Elasticsearch(
            hosts=[f"http://{host}:{port}"],
            verify_certs=False,
            request_timeout=120,  # 2 minutes timeout
            max_retries=3,
            retry_on_timeout=True
        )
    except Exception as e:
        print(f"❌ Failed to create Elasticsearch client: {e}")
        return False
    
    # Check connection
    print(f"📍 Connecting to Elasticsearch at {host}:{port}...")
    try:
        if not es.ping():
            print("❌ Elasticsearch is not responding")
            return False
        print("✅ Elasticsearch is reachable")
    except ConnectionError as e:
        print(f"❌ Connection failed: {e}")
        return False
    
    # Check if index exists
    print(f"\n📁 Checking if index '{index_name}' exists...")
    try:
        if not es.indices.exists(index=index_name):
            print(f"ℹ️  Index '{index_name}' does not exist")
            return True
        else:
            # Get index stats before deletion
            stats = es.indices.stats(index=index_name)
            index_stats = stats['indices'][index_name]
            doc_count = index_stats['total']['docs']['count']
            size_mb = index_stats['total']['store']['size_in_bytes'] / (1024*1024)
            
            print(f"📊 Index found:")
            print(f"   📄 Documents: {doc_count:,}")
            print(f"   💾 Size: {size_mb:.2f} MB")
            
    except Exception as e:
        print(f"⚠️  Could not check index status: {e}")
        return False
    
    # Confirm deletion
    print(f"\n⚠️  About to delete index '{index_name}'")
    print(f"   This will permanently remove all {doc_count:,} documents")
    
    try:
        confirm = input("   Are you sure? (y/n): ").strip().lower()
        if confirm not in ['y']:
            print("❌ Deletion cancelled")
            return True
    except KeyboardInterrupt:
        print("\n❌ Deletion cancelled")
        return True
    
    # Delete index with retry logic
    max_retries = 3
    retry_delay = 10
    
    for attempt in range(max_retries):
        try:
            print(f"\n🗑️  Attempting to delete index (attempt {attempt + 1}/{max_retries})...")
            
            # Delete with extended timeout
            es.indices.delete(
                index=index_name,
                request_timeout=180,  # 3 minutes
                master_timeout="180s"  # 3 minutes with time unit
            )
            
            print(f"✅ Successfully deleted index '{index_name}'")
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "process_cluster_event_timeout_exception" in error_msg:
                if attempt < max_retries - 1:
                    print(f"⚠️  Timeout error on attempt {attempt + 1}/{max_retries}")
                    print(f"   Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"❌ Failed to delete index after {max_retries} attempts")
                    print(f"   Error: {error_msg}")
                    print("\n💡 Possible solutions:")
                    print("   1. Wait a few minutes and try again")
                    print("   2. Restart Elasticsearch: docker restart <container_name>")
                    print("   3. Check if other operations are using the index")
                    print("   4. Check available disk space")
                    return False
            else:
                print(f"❌ Unexpected error: {error_msg}")
                return False
    
    return False

def main():
    """Main function"""
    success = delete_index()
    
    if success:
        print("\n✅ Index deletion completed successfully")
    else:
        print("\n❌ Index deletion failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 