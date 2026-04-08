import re


def clean_text(text: str) -> str:
    """
    Apply text cleaning transformations.

    This mirrors the notebook's `clean_text` function so the TF-IDF features
    match what the model was trained on.
    """
    text = str(text)

    # Remove URLs
    text = re.sub(r"https?://\\S+|www\\.\\S+", "", text)
    text = re.sub(r"\\S+\\.com\\S*", "", text)
    text = re.sub(r"\\S+\\.co/\\S*", "", text)
    text = re.sub(r"pic\\.\\S+", "", text)

    # Remove @mentions
    text = re.sub(r"@\\w+", "", text)

    # Remove special tokens/placeholders present in the dataset
    text = re.sub(r"<unk>", "", text)
    text = re.sub(r"RhandlerR", "", text)
    text = re.sub(r"RhttpR", "", text)

    # Normalize whitespace
    text = re.sub(r"\\s+", " ", text).strip()
    return text

