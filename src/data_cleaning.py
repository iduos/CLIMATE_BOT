# src/data_cleaning.py
#
# By Ian Drumm, The Univesity of Salford, UK.
#
import re
from typing import List, Dict

# Simple URL detector: matches http(s), www., or bare domains like example.com/foo
URL_RE = re.compile(
    r'(?i)\b(?:https?://|www\.)\S+|\b[a-z0-9.-]+\.(?:com|net|org|io|co|uk|edu|gov|me|info|news)(?:/\S*)?'
)

def _has_url(text: str) -> bool:
    return bool(URL_RE.search(text))

def clean_reddit_data(data: List[Dict]) -> List[Dict]:
    """
    Filters out unusable comments (e.g., '[removed]', '[deleted]', bot messages, incomplete),
    and removes entries where posts or comments contain URLs.
    """
    filtered_data = []
    unusable_comment_markers = {"[removed]", "[deleted]"}
    # Common text fields that might hold post/comment content in your data
    text_fields_to_check = ("comment", "post", "title", "selftext", "body")

    removed_count = 0

    for entry in data:
        comment_content = entry.get("comment")

        is_usable = True
        if comment_content is None:
            is_usable = False
        elif isinstance(comment_content, str):
            normalized_comment = comment_content.strip().lower()
            if normalized_comment in unusable_comment_markers:
                is_usable = False
            elif "automoderator" in normalized_comment and "i am a bot" in normalized_comment:
                is_usable = False
            elif len(comment_content.strip()) < 10:
                is_usable = False
            elif "..." in comment_content.strip() and len(comment_content.strip()) < 20:
                is_usable = False
        else:
            is_usable = False

        # New: remove if any relevant text field contains a URL
        if is_usable:
            # If there's an explicit URL field and it's non-empty, drop it
            if isinstance(entry.get("url"), str) and entry["url"].strip():
                is_usable = False
            else:
                for field in text_fields_to_check:
                    val = entry.get(field)
                    if isinstance(val, str) and _has_url(val):
                        is_usable = False
                        break

        if is_usable:
            filtered_data.append(entry)
        else:
            removed_count += 1

    print(f"Cleaned data of {removed_count} items with unusable comments or URLs.")
    return filtered_data
