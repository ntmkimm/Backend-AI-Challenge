import os
import time
import sys
from es_module.services.es_service2 import Service

def check_disk_space():
    """Check available disk space and warn if low"""
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024**3)
        used_percent = (used / total) * 100
        
        print(f"💾 Disk Usage: {used_percent:.1f}% used, {free_gb:.1f} GB free")
        
        if used_percent > 90:
            print("⚠️  WARNING: Disk usage is very high (>90%)")
            print("   This may cause Elasticsearch timeouts and performance issues")
            print("   Consider freeing up disk space before proceeding")
            return False
        elif used_percent > 80:
            print("⚠️  WARNING: Disk usage is high (>80%)")
            print("   Consider freeing up disk space for better performance")
        else:
            print("✅ Disk space looks good")
        return True
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
        return True

def check_elasticsearch_connection():
    """Check if Elasticsearch is reachable"""
    try:
        service = Service()
        if service.client.check_connection():
            print("✅ Elasticsearch connection successful")
            return True
        else:
            print("❌ Elasticsearch is not responding")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to Elasticsearch: {e}")
        return False

def force_delete_index(service):
    """Force delete index with extended timeout and retry logic"""
    print("🗑️  Force deleting existing index...")
    
    max_retries = 3
    retry_delay = 15
    
    for attempt in range(max_retries):
        try:
            print(f"   Attempt {attempt + 1}/{max_retries}...")
            
            # Use extended timeout for deletion with proper options
            service.client.es.indices.delete(
                index=service.client.config.INDEX_NAME,
                **service.client.es.options(
                    request_timeout=300,  # 5 minutes
                    master_timeout="300s"  # 5 minutes
                ).params()
            )
            print("✅ Index deleted successfully!")
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "process_cluster_event_timeout_exception" in error_msg:
                if attempt < max_retries - 1:
                    print(f"   ⚠️  Timeout error, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"   ❌ Failed to delete index after {max_retries} attempts")
                    print(f"   Error: {error_msg}")
                    return False
            elif "index_not_found_exception" in error_msg:
                print("✅ Index doesn't exist (already deleted)")
                return True
            else:
                print(f"   ❌ Unexpected error: {error_msg}")
                return False
    
    return False

def initialize_index_with_retry(service, max_retries=3):
    """Initialize index with multiple retry strategies"""
    
    # Strategy 1: Try normal recreation
    print("🔄 Strategy 1: Normal index recreation...")
    for attempt in range(max_retries):
        try:
            service.initialize_index(recreate=True)
            print("✅ Index initialized successfully!")
            return True
        except Exception as e:
            error_msg = str(e)
            if "process_cluster_event_timeout_exception" in error_msg:
                if attempt < max_retries - 1:
                    delay = 10 * (2 ** attempt)  # Exponential backoff
                    print(f"⚠️  Timeout error on attempt {attempt + 1}/{max_retries}")
                    print(f"   Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"❌ Failed with timeout after {max_retries} attempts")
                    break
            else:
                print(f"❌ Unexpected error: {error_msg}")
                break
    
    return False

def display_results(results, max_display=20):
    """Display search results with pagination"""
    if not results:
        print("   No results found")
        return
    
    total_results = len(results)
    print(f"   Found {total_results} results:")
    
    # Display first batch
    for i, result in enumerate(results[:max_display], 1):
        video_id, frame_id, score, filepath = result
        print(f"   {i:3d}. Score: {score:.3f} | Video: {video_id} | Frame: {frame_id}")
        print(f"       File: {filepath}")
    
    # If there are more results, show summary
    if total_results > max_display:
        remaining = total_results - max_display
        print(f"   ... and {remaining} more results")
        print(f"   (Showing first {max_display} results)")

def search_with_options(service, query, search_type="combined", max_results=1000, display_limit=20):
    """Search with customizable options"""
    print(f"   🔍 {search_type.title()} search (max {max_results} results):")
    try:
        results = service.search(query, size=max_results, search_type=search_type)
        display_results(results, display_limit)
        return results
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return []

def main():
    print("🚀 Starting Advanced  Dataset Upload and Test")
    print("=" * 60)
    
    # Cấu hình kết nối qua biến môi trường
    os.environ["ELASTICSEARCH_HOST"] = "elasticsearch"
    os.environ["ELASTICSEARCH_PORT"] = "9200"
    
    # Set dataset path
    dataset_path = os.path.join(os.path.dirname(__file__), "../../dataset")
    os.environ["DATASET_PATH"] = dataset_path
    
    # Check prerequisites
    print("\n📋 Checking prerequisites...")
    
    # Check disk space
    try:
        skip_disk_check = input("   Skip disk space check? (y/n) [default: n]: ").strip().lower()
        if skip_disk_check not in ['y', 'yes']:
            disk_ok = check_disk_space()
            if not disk_ok:
                print("\n⚠️  WARNING: Disk space is very low!")
                print("   This may cause timeouts and performance issues.")
                print("   Consider freeing up space for better performance.")
                
                try:
                    continue_anyway = input("   Continue anyway? (y/n) [default: n]: ").strip().lower()
                    if continue_anyway not in ['y', 'yes']:
                        print("❌ Operation cancelled due to low disk space.")
                        return
                    else:
                        print("🔄 Continuing with low disk space...")
                except KeyboardInterrupt:
                    print("\n❌ Operation cancelled by user")
                    return
        else:
            print("⏭️  Skipping disk space check...")
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        return
    
    # Check dataset path
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path does not exist: {dataset_path}")
        print("💡 Make sure the dataset directory exists and contains the  data")
        return
    
    print(f"✅ Dataset path exists: {dataset_path}")
    
    # Check Elasticsearch connection
    if not check_elasticsearch_connection():
        print("\n❌ Cannot connect to Elasticsearch. Please check:")
        print("   1. Elasticsearch is running: docker ps")
        print("   2. Network connectivity to 172.17.0.3:9200")
        print("   3. Elasticsearch container is healthy")
        return
    
    # Khởi tạo service
    print("\n🔧 Initializing  service...")
    service = Service()
    print("database name: ", service.client.config.INDEX_NAME)
    try:
        # 1. Khởi tạo index với retry logic
        print("\n📁 Initializing Elasticsearch index...")
        
        # Check if index exists and ask user
        try:
            if service.client.es.indices.exists(index=service.client.config.INDEX_NAME):
                print(f"📁 Index '{service.client.config.INDEX_NAME}' already exists")
                print("💡 Options:")
                print("   1. Force delete and recreate (recommended for fresh start)")
                print("   2. Use existing index (faster, but may have old data)")
                print("   3. Skip index initialization")
                
                try:    
                    choice = input("   Choose option (1/2/3) [default: 1]: ").strip()
                    if choice == "" or choice == "1":
                        print("🗑️  Force deleting existing index...")
                        if force_delete_index(service):
                            service.initialize_index(recreate=False)
                            print("✅ Index recreated successfully!")
                        else:
                            print("❌ Failed to delete index.")
                            print("💡 Would you like to:")
                            print("   1. Try again")
                            print("   2. Use existing index")
                            print("   3. Exit")
                            
                            try:
                                retry_choice = input("   Choose option (1/2/3) [default: 2]: ").strip()
                                if retry_choice == "1":
                                    print("🔄 Retrying index deletion...")
                                    if force_delete_index(service):
                                        service.initialize_index(recreate=False)
                                        print("✅ Index recreated successfully!")
                                    else:
                                        print("❌ Still failed. Using existing index...")
                                        service.initialize_index(recreate=False)
                                        print("✅ Using existing index successfully!")
                                elif retry_choice == "3":
                                    print("❌ Exiting...")
                                    return
                                else:  # default or "2"
                                    print("📁 Using existing index...")
                                    service.initialize_index(recreate=False)
                                    print("✅ Using existing index successfully!")
                            except KeyboardInterrupt:
                                print("\n❌ Operation cancelled by user")
                                return
                    elif choice == "2":
                        print("📁 Using existing index...")
                        service.initialize_index(recreate=False)
                        print("✅ Using existing index successfully!")
                    elif choice == "3":
                        print("⏭️  Skipping index initialization...")
                    else:
                        print("❌ Invalid choice. Exiting.")
                        return
                except KeyboardInterrupt:
                    print("\n❌ Operation cancelled by user")
                    return
            else:
                print("📁 Creating new index...")
                if not initialize_index_with_retry(service):
                    print("\n❌ Failed to initialize index. Exiting.")
                    return
        except Exception as e:
            print(f"⚠️  Error checking index status: {e}")
            print("🔄 Proceeding with automatic initialization...")
            if not initialize_index_with_retry(service):
                print("\n❌ Failed to initialize index. Exiting.")
                return

        # 2. Upload dataset
        print("\n📤 Uploading dataset...")
        print(f"   Dataset path: {dataset_path}")
        service.index_dataset(dataset_path)
        print("✅ Dataset upload completed!")

        # 3. Test tìm kiếm với tùy chọn nâng cao
        print("\n🔍 Testing search functionality with unlimited results...")
        
        # Cấu hình tìm kiếm
        max_results = 1000  # Số lượng kết quả tối đa
        display_limit = 1000  # Số lượng kết quả hiển thị
        
        test_queries = [
            "đối tượng mua bán trái phép",
            "hành vi vi phạm", 
            "xử lý vi phạm",
            "phạt tiền",
            "tạm giữ",
            "thu giữ"
        ]
        
        for query in test_queries:
            print(f"\n🔎 Testing query: '{query}'")
            print("-" * 50)
            
            # Test với exact match
            exact_results = search_with_options(
                service, query, "exact", max_results, display_limit
            )
            
            # Test với fuzzy match
            fuzzy_results = search_with_options(
                service, query, "fuzzy", max_results, display_limit
            )
            
            # Test với combined match
            combined_results = search_with_options(
                service, query, "combined", max_results, display_limit
            )
            
            # Tổng kết cho query này
            print(f"\n📊 Summary for '{query}':")
            print(f"   Exact matches: {len(exact_results)}")
            print(f"   Fuzzy matches: {len(fuzzy_results)}")
            print(f"   Combined matches: {len(combined_results)}")

        print("\n" + "=" * 60)
        print("✅ Advanced upload and test completed successfully!")
        print("🎉 Your  search system is ready with unlimited results!")
        print("\n💡 Tips:")
        print("   - You can modify max_results and display_limit variables")
        print("   - Add more test queries to the test_queries list")
        print("   - Results are sorted by relevance score")

    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Check the error message above for troubleshooting")

if __name__ == "__main__":
    main() 