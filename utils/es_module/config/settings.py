from typing import Dict
import os

class ElasticsearchConfig:
    """Configuration settings for Elasticsearch"""
    
    HOST: str = os.getenv("ELASTICSEARCH_HOST", "elasticsearch")  # Default to Docker container IP
    PORT: int = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
    USERNAME: str = os.getenv("ELASTICSEARCH_USER", "")  # Empty if no auth
    PASSWORD: str = os.getenv("ELASTICSEARCH_PASS", "")  # Empty if no auth
    
    INDEX_NAME: str = "aic2025"
    
    @classmethod
    def get_connection_config(cls) -> Dict:
        """Get Elasticsearch connection configuration"""
        config = {
            "hosts": [f"http://{cls.HOST}:{cls.PORT}"],
            "verify_certs": False,
            "request_timeout": 300,
            "max_retries": 3,
            "retry_on_timeout": True
        }
        
        if cls.USERNAME and cls.PASSWORD:
            config["basic_auth"] = (cls.USERNAME, cls.PASSWORD)
            
        return config
    
    @classmethod
    def get_index_settings(cls) -> Dict:
        """Get Elasticsearch index settings"""
        return {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "index.mapping.ignore_malformed": True,
                "analysis": {
                    "analyzer": {
                        "vietnamese": {
                            "tokenizer": "standard",
                            "filter": ["lowercase", "asciifolding"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "video_id": {"type": "keyword"},
                    "frame_id": {"type": "keyword"},
                    "ocr_text": {
                        "type": "text",
                        "analyzer": "vietnamese",
                        "fields": {
                            "raw": {"type": "keyword"},
                            "exact": {"type": "text", "analyzer": "standard"}
                        }
                    },
                    "thumbnail_ocr": {
                        "type": "text",
                        "analyzer": "vietnamese",
                        "fields": {
                            "raw": {"type": "keyword"},
                            "exact": {"type": "text", "analyzer": "standard"}
                        }
                    }
                }
            }
        } 