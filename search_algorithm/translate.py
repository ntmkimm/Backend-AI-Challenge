import requests

def google_translate(text, source_lang='auto', target_lang='en'):
    url = "https://translate.googleapis.com/translate_a/single"
    params = {
        'client': 'gtx',
        'sl': source_lang,
        'tl': target_lang,
        'dt': 't',
        'q': text
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        # Response format: [[["translated text", "original text", null, null, ...]], null, "source language"]
        result = response.json()
        translated_text = ''.join([item[0] for item in result[0]])
        detected_source = result[2]  # Detected source language
        return translated_text, detected_source
    else:
        raise Exception(f"Translation failed with status code {response.status_code}")

# Example
text = "Xin chào, bạn khỏe không?"
translated, detected_lang = google_translate(text, source_lang='auto', target_lang='en')
print(f"Translated: {translated} (Detected: {detected_lang})")
