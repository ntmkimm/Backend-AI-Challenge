# api/dependencies/services.py
from typing import Optional, Dict
import os
import threading
import asyncio

from services.clip_service import CLIPService
from services.milvus_service import MilvusService
from services.polar_service import PolarService
from services.beit3_service import BEiT3Service
from services.siglip2_service import SigLIP2Service
from services.interval_service import IntervalService

from config.settings import (
    OPENCLIP_MILVUS,
    BEIT3_MILVUS,
    SIGLIP2_MILVUS,
    DEVICE_0,
    DEVICE_1,
)

class ServiceManager:
    def __init__(self):
        self.clip_service: Optional[CLIPService] = None
        self.redis_service = None
        self.polar_service: Optional[PolarService] = None
        self.beit3_service: Optional[BEiT3Service] = None
        self.siglip2_service: Optional[SigLIP2Service] = None
        
        self.milvus_services: Dict[str, Optional[MilvusService]] = {
            OPENCLIP_MILVUS: None,
            BEIT3_MILVUS: None,
            SIGLIP2_MILVUS: None,
        }
        self.interval_service: Optional[IntervalService] = None
        
        self.lock = threading.Lock()
        self.async_lock = asyncio.Lock()

    def get_clip_service(self, device=DEVICE_0) -> CLIPService:
        if self.clip_service is None:
            with self.lock:
                if self.clip_service is None:
                    self.clip_service = CLIPService(device=device)
        return self.clip_service

    def get_milvus_service_with_collection(self, collection_name: str) -> MilvusService:
        if self.milvus_services[collection_name] is None:
            with self.lock:
                if self.milvus_services[collection_name] is None:
                    self.milvus_services[collection_name] = MilvusService(collection_name=collection_name)
        return self.milvus_services[collection_name]

    def get_milvus_services(self) -> Dict[str, MilvusService]:
        # Ensure all are initialized (including SIGLIP2_MILVUS)
        # self.get_milvus_service_with_collection(OPENCLIP_MILVUS)
        # self.get_milvus_service_with_collection(BEIT3_MILVUS)
        # self.get_milvus_service_with_collection(SIGLIP2_MILVUS)
        # type: ignore[return-value]
        return self.milvus_services  # now filled

    async def get_redis_service(self):
        from services.redis_service import RedisService
        
        if self.redis_service is None:
            async with self.async_lock:
                if self.redis_service is None:
                    instance = RedisService()
                    # async_init() must return a RedisService (common pattern)
                    self.redis_service = await instance.async_init()
        return self.redis_service

    def get_polar_service(self) -> PolarService:
        if self.polar_service is None:
            with self.lock:
                if self.polar_service is None:
                    self.polar_service = PolarService()
        return self.polar_service

    def get_interval_service(self) -> IntervalService:
        if self.interval_service is None:
            with self.lock:
                if self.interval_service is None:
                    self.interval_service = IntervalService()
        return self.interval_service

    def get_beit3_service(self, device=DEVICE_0) -> BEiT3Service:
        if self.beit3_service is None:
            with self.lock:
                if self.beit3_service is None:
                    self.beit3_service = BEiT3Service(device=device)
        return self.beit3_service

    def get_siglip2_service(self, device=DEVICE_1) -> SigLIP2Service:
        if self.siglip2_service is None:
            with self.lock:
                if self.siglip2_service is None:
                    self.siglip2_service = SigLIP2Service(device=device)
        return self.siglip2_service


# Global singleton
service_manager = ServiceManager()

# === Dependency providers used by FastAPI ===

def get_milvus_service_with_collection(collection_name: str) -> MilvusService:
    return service_manager.get_milvus_service_with_collection(collection_name)

def get_milvus_services() -> Dict[str, MilvusService]:
    return service_manager.get_milvus_services()

async def get_redis_service():
    return await service_manager.get_redis_service()

def get_polar_service() -> PolarService:
    return service_manager.get_polar_service()

def get_interval_service() -> IntervalService:
    return service_manager.get_interval_service()

def get_clip_service(device=DEVICE_0) -> CLIPService:
    return service_manager.get_clip_service(device=device)

def get_beit3_service(device=DEVICE_0) -> BEiT3Service:
    return service_manager.get_beit3_service(device=device)

def get_siglip2_service(device=DEVICE_1) -> SigLIP2Service:
    return service_manager.get_siglip2_service(device=device)

# === Agent dependencies ===

async def get_agent_controller():
    """
    Async dependency for routers that need the agent.
    """
    from services.agents.controller import AgentController
    from services.agents.runtime import AgentRuntime
    from services.agents import tools as agent_tools
    redis = await service_manager.get_redis_service()
    # If your AgentRuntime needs polar_service too, wire it here:
    # polar = service_manager.get_polar_service()

    # Pull provider/model/key from env or settings
    provider = os.getenv("AGENT_PROVIDER", "openai")  # openai|anthropic|gemini
    api_key = (
        os.getenv("OPENAI_API_KEY")
        if provider == "openai" else
        os.getenv("ANTHROPIC_API_KEY") if provider == "anthropic" else
        os.getenv("GEMINI_API_KEY")
    )
    model = os.getenv("AGENT_MODEL")  # optional

    controller = AgentController(
        provider=provider,
        api_key=api_key,
        model=model,
        redis_service=redis,
    )

    # Initialize tools’ runtime (shared with agent tool-calling if you use it)
    # If your AgentRuntime requires polar: AgentRuntime(redis, polar)
    agent_tools.runtime = AgentRuntime(redis_service=redis)  # or AgentRuntime(redis, polar)

    return controller
async def get_quick_agent_controller():
    """
    Async dependency for routers that need the quick agent.
    """
    from services.agents import tools as agent_tools
    from services.agents.runtime import AgentRuntime
    from services.agents.quick_controller import QuickAgentController
    redis = await service_manager.get_redis_service()

    # Pull provider/model/key from env or settings
    provider = os.getenv("AGENT_PROVIDER", "openai")  # openai|anthropic|gemini
    api_key = (
        os.getenv("OPENAI_API_KEY")
        if provider == "openai" else
        os.getenv("ANTHROPIC_API_KEY") if provider == "anthropic" else
        os.getenv("GEMINI_API_KEY")
    )
    model = os.getenv("AGENT_MODEL")  # optional

    controller = QuickAgentController(
        provider=provider,
        api_key=api_key,
        model=model,
        redis_service=redis,
    )
    agent_tools.runtime = AgentRuntime(redis_service=redis)  # or AgentRuntime(redis, polar)

    return controller
# === Runtime dependency (add this) ===
from services.agents.runtime import AgentRuntime
_runtime_singleton: AgentRuntime 


async def get_runtime():
    from services.agents.controller import AgentController
    from services.agents import tools as agent_tools
    
    global _runtime_singleton
    if _runtime_singleton is None:
        redis = await service_manager.get_redis_service()
        polar = service_manager.get_polar_service()  # if you want frame_information to include objects
        _runtime_singleton = AgentRuntime(redis_service=redis, polar_service=polar)
        # keep tools runtime in sync (optional, for tool-calling agents)
        agent_tools.runtime = _runtime_singleton
    return _runtime_singleton
