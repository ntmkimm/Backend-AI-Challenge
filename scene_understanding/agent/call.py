from pydantic import BaseModel
from typing import List, Optional
from langchain_openai import ChatOpenAI
import os
import time

# Define query structure
class Query(BaseModel):
    text: str        
    asr: str         
    ocr: str         
    origin: str  
    obj: List[str]
    lang: str
    image: Optional[str] = ""

# Load API key
api_key = os.environ.get("OPENAI_API_KEY")

# Initialize GPT-4o mini
# llm = ChatOpenAI(
#     openai_api_key=api_key,
#     openai_api_base="https://api.openai.com/v1",
#     model="gpt-4o",
#     temperature=0.6,
#     top_p=0.95,
# )

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Long input query
full_query = (
    "Trận bóng đá giữa đội Uzbekistan và đội Triều Tiên. "
    "Một đội trong trang phục toàn trắng và đội còn lại trong trang phục toàn xanh dương. "
    "Đây là thời điểm đội Triều Tiên được hưởng quả phạt đền 11 mét. "
    "Hỏi lúc thực hiện quả phạt đền có bao nhiêu cầu thủ Uzbekistan đang ở trong khung hình?"
)

# Prompt
prompt = f"""
Given the following long query, split it into a list of structured sub-queries for image/video analysis, following this Python Pydantic model:

class Query(BaseModel):
    text: str        
    asr: str         
    ocr: str         
    origin: str  
    obj: List[str]
    lang: str
    image: Optional[str] = ""

- text: the main question or recognition request in English (may be translated from the original)
- asr: automatic speech recognition text if available (leave blank if not)
- ocr: optical character recognition text if available (leave blank if not)
- origin: the original question or sentence (in the original language)
- obj: list of objects to recognize or detect (use COCO-like object categories, e.g., "person", "train", "uniform", "penalty", "frame", etc.)
- lang: language code, e.g., 'vi' for Vietnamese, 'en' for English
- image: image path if available (leave blank if not)

For each sub-query, fill in all fields, use COCO-like object categories for 'obj'. Output the result as a Python list of dicts.

Input:
{full_query}
"""

# Run
start_time = time.time()
messages = [{"role": "user", "content": prompt}]
response = llm.invoke(messages)
print(response.content)
end_time = time.time()

print("Time call:", end_time - start_time)


# Multimodal invocation with gemini-pro-vision
# message = HumanMessage(
#     content=[
#         {
#             "type": "text",
#             "text": "What's in this image?",
#         },
#         {"type": "image_url", "image_url": "https://picsum.photos/seed/picsum/200/300"},
#     ]
# )
# result = llm.invoke([message])
# print(result.content)