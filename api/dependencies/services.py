from services.clip_service import CLIPService
from services.milvus_service import MilvusService
from services.polar_service import PolarService
from services.beit3_service import BEiT3Service
from services.paraphrase_service import ParaphraseService
from config.settings import OPENCLIP_BATCH1, BEIT3_BATCH1, DEVICE_0, DEVICE_1
import threading
import asyncio

class ServiceManager:
    def __init__(self):
        self.clip_service = None
        self.redis_service = None
        self.polar_service = None
        self.paraphrase_service = None
        self.beit3_service = None
        self.milvus_services = {
            OPENCLIP_BATCH1: None,
            BEIT3_BATCH1: None,
        }
        
        self.milvus_beit3_batch1 = None
        self.milvus_openclip_batch1 = None
        
        self.lock = threading.Lock()  # Lock to handle concurrent initialization
        self.async_lock = asyncio.Lock()

    def get_clip_service(self, device):
        if self.clip_service is None:
            with self.lock:
                if self.clip_service is None:  # Double-check locking
                    self.clip_service = CLIPService(device=device)
        return self.clip_service
    
    def get_milvus_service_with_collection(self, collection_name) -> MilvusService:
        if self.milvus_services[collection_name] is None:
            with self.lock:
                if self.milvus_services[collection_name] is None:
                    self.milvus_services[collection_name] = MilvusService(collection_name=collection_name)
        return self.milvus_services[collection_name]
    
    def get_milvus_services(self) -> dict:
        # self.get_milvus_service_with_collection(collection_name=OPENCLIP_BATCH1)
        # self.get_milvus_service_with_collection(collection_name=BEIT3_BATCH1)
        return self.milvus_services

    async def get_redis_service(self):
        from services.redis_service import RedisService
        if self.redis_service is None:
            async with self.async_lock:
                if self.redis_service is None:
                    instance = RedisService()
                    self.redis_service = await instance.async_init()
        return self.redis_service

    def get_polar_service(self):
        if self.polar_service is None:
            with self.lock:
                if self.polar_service is None:
                    self.polar_service = PolarService()
        return self.polar_service
    
    def get_paraphrase_service(self):
        if self.paraphrase_service is None:
            with self.lock:
                if self.paraphrase_service is None:
                    self.paraphrase_service = ParaphraseService()
        return self.paraphrase_service
    
    def get_beit3_service(self, device):
        if self.beit3_service is None:
            with self.lock:
                if self.beit3_service is None:
                    self.beit3_service = BEiT3Service(device=device)
        return self.beit3_service

# Initialize service manager globally
service_manager = ServiceManager()

# Replace original get_* functions with the methods from ServiceManager
def get_clip_service():
    return service_manager.get_clip_service(device=DEVICE_0)

def get_milvus_service_with_collection(collection_name):
    return service_manager.get_milvus_service_with_collection(collection_name=collection_name)

def get_mivus_services():
    return service_manager.get_milvus_services()

async def get_redis_service():
    return await service_manager.get_redis_service()

def get_polar_service():
    return service_manager.get_polar_service()

def get_paraphrase_service():
    return service_manager.get_paraphrase_service()

def get_beit3_service():
    return service_manager.get_beit3_service(device=DEVICE_1)
