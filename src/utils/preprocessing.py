import re
from typing import Optional


def clean_text(text: str, lowercase: bool = True) -> str:
    if not isinstance(text, str):
        return ""

    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    if lowercase:
        text = text.lower()

    return text


def remove_special_characters(text: str, keep_punctuation: bool = True) -> str:
    if not isinstance(text, str):
        return ""

    if keep_punctuation:
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\'-]', '', text)
    else:
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    return text


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def preprocess_comment(
    text: str,
    clean: bool = True,
    lowercase: bool = True,
    remove_special_chars: bool = False,
    normalize: bool = False
) -> str:
    if not isinstance(text, str):
        return ""

    if clean:
        text = clean_text(text, lowercase=lowercase)
    elif lowercase:
        text = text.lower()

    if remove_special_chars:
        text = remove_special_characters(text)

    if normalize:
        text = normalize_text(text)

    return text


def batch_preprocess(
    texts: list[str],
    clean: bool = True,
    lowercase: bool = True,
    remove_special_chars: bool = False,
    normalize: bool = False
) -> list[str]:
    return [
        preprocess_comment(text, clean, lowercase, remove_special_chars, normalize)
        for text in texts
    ]

