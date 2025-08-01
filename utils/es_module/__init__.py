"""
Elasticsearch module for OCR search functionality
"""

from typing import List
from .services.es_service2 import Service

# Create singleton service instance
_service = None

def get_service() -> Service:
    """Get or create service singleton"""
    global _service
    if _service is None:
        _service = Service()
    return _service

def search_by_ocr(query: str, size: int = 10) -> List[str]:
    """
    Search for frames containing OCR text.

    Args:
        query (str): The OCR text to search for.
        size (int, optional): Số lượng kết quả tối đa trả về (mặc định là 10).

    Returns:
        List[str]: Danh sách các frame chứa văn bản OCR phù hợp.
    """
    service = get_service()
    return service.search(query, size, fields=['ocr_text'])

def search_by_asr(query: str, size: int = 10) -> List[str]:
    """
    Search for frames containing ASR text.

    Args:
        query (str): The ASR text to search for.
        size (int, optional): Số lượng kết quả tối đa trả về (mặc định là 10).

    Returns:
        List[str]: Danh sách các frame chứa văn bản OCR phù hợp.
    """
    service = get_service()
    return service.search(query, size, fields=['asr_text'])

def search_by_ocr_asr(query: str, size: int = 10) -> List[str]:
    """
    Search for frames containing OCR text.

    Args:
        query (str): The OCR text to search for.
        size (int, optional): Số lượng kết quả tối đa trả về (mặc định là 10).

    Returns:
        List[str]: Danh sách các frame chứa văn bản OCR phù hợp.
    """
    service = get_service()
    return service.search(query, size, fields=['ocr_text', 'asr_text'])