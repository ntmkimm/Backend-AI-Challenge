from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os

# Initialize client with your API key
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Input text to paraphrase
text = "Cảnh một bạn nam đang đứng trước sản phẩm của mình trong một cuộc thi. Sau đó bạn nam này trả lời phỏng vấn bằng ngôn ngữ ký hiệu.."

# Call the model for paraphrasing
response = client.chat.completions.create(
    model="gpt-4o",   # you can switch to another model if needed
    messages=[
        {"role": "system", "content": """
         Bạn là agent hiểu về cách parahrase và dịch sang tiếng anh đoạn text sau để clip, siglip2 và beit3 hiểu.
         Đảm bảo trả về dạng json như sau:
         {
             "clip": "paraphrase đoạn text thành tiếng anh để clip hiểu",
             "siglip2": "paraphrase đoạn text thành tiếng anh để siglip2 hiểu",
             "beit3": "paraphrase đoạn text thành tiếng anh để beit3 hiểu"
         }
         """},
        {"role": "user", "content": f"TEXT: {text}"}
    ],
)

import json
paraphrased = response.choices[0].message.content
jsons = json.loads(paraphrased)

print(jsons)
