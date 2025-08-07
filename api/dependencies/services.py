from services.clip_service import CLIPService
from services.milvus_service import MilvusService
from services.redis_service import RedisService
from services.polar_service import PolarService

import threading

class ServiceManager:
    def __init__(self):
        self.clip_service = None
        self.milvus_service = None
        self.redis_service = None
        self.polar_service = None
        self.lock = threading.Lock()  # Lock to handle concurrent initialization

    def get_clip_service(self):
        if self.clip_service is None:
            with self.lock:
                if self.clip_service is None:  # Double-check locking
                    self.clip_service = CLIPService()
        return self.clip_service

    def get_milvus_service(self):
        if self.milvus_service is None:
            with self.lock:
                if self.milvus_service is None:
                    self.milvus_service = MilvusService()
        return self.milvus_service

    def get_redis_service(self):
        if self.redis_service is None:
            with self.lock:
                if self.redis_service is None:
                    self.redis_service = RedisService()
        return self.redis_service

    def get_polar_service(self):
        if self.polar_service is None:
            with self.lock:
                if self.polar_service is None:
                    self.polar_service = PolarService()
        return self.polar_service

# Initialize service manager globally
service_manager = ServiceManager()

# Replace original get_* functions with the methods from ServiceManager
def get_clip_service():
    return service_manager.get_clip_service()

def get_milvus_service():
    return service_manager.get_milvus_service()

def get_redis_service():
    return service_manager.get_redis_service()

def get_polar_service():
    return service_manager.get_polar_service()
