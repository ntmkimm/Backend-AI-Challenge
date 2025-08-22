import os
import time
import sys
from es_module.services.es_service2 import Service

def check_elasticsearch_connection():
    """Check if Elasticsearch is reachable"""
    try:
        service = Service()
        if service.client.check_connection():
            print("Elasticsearch connection successful")
            return True
        else:
            print("Elasticsearch is not responding")
            return False
    except Exception as e:
        print(f"Failed to connect to Elasticsearch: {e}")
        return False


def main():
    # Cấu hình kết nối qua biến môi trường
    os.environ["ELASTICSEARCH_HOST"] = "elasticsearch"
    os.environ["ELASTICSEARCH_PORT"] = "9200"
    
    # Set dataset path
    dataset_path = os.path.join(os.path.dirname(__file__), "../../dataset/full")
    os.environ["DATASET_PATH"] = dataset_path
    
    # Check prerequisites
    print("\n📋 Checking prerequisites...")
    
    # Check dataset path
    if not os.path.exists(dataset_path):
        print("💡 Make sure the dataset directory exists and contains the  data")
        return
    
    print(f"Dataset path exists: {dataset_path}")
    
    # Check Elasticsearch connection
    if not check_elasticsearch_connection():
        print("\nCannot connect to Elasticsearch")
        return
    
    print("\n🔧 Initializing  service...")
    service = Service()

    # 2. Upload dataset
    print("\n📤 Uploading dataset...")
    print(f"   Dataset path: {dataset_path}")
    # service.initialize_index()
    service.index_dataset(dataset_path)
    
    # curl -X PUT "http://elasticsearch:9200/_cluster/settings" -H 'Content-Type: application/json' -d '{ 
    #   "transient": {
    #     "cluster.routing.allocation.disk.watermark.low": "98%",
    #     "cluster.routing.allocation.disk.watermark.high": "100%", 
    #     "cluster.routing.allocation.disk.watermark.flood_stage": "100%" 
    #   }
    # }'

    print("✅ Dataset upload completed!")

if __name__ == "__main__":
    main() 