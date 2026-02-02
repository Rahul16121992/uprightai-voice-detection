from langdetect import detect, LangDetectException

LANG_MAP = {
    "hi": "Hindi",
    "en": "English",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam"
}

def detect_language_from_text(text: str):
    try:
        code = detect(text)
        return LANG_MAP.get(code, "Unknown")
    except LangDetectException:
        return "Unknown"
