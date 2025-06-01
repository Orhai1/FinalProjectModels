import os
import easyocr

# Load keywords from text files
def load_keywords_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

hamas_keywords = load_keywords_from_file('hamas_keywords.txt')
fatah_keywords = load_keywords_from_file('fatah_keywords.txt')

# Initialize EasyOCR
reader = easyocr.Reader(['ar', 'en'], gpu=True)

def extract_text_from_frame(path):
    results = reader.readtext(path)
    combined_text = ' '.join([text for _, text, _ in results])
    return combined_text

def check_text_sources(frames_dir, description, username):
    flags = {
        "hamas_text_ocr": 0,
        "fatah_text_ocr": 0,
        "hamas_text_description": 0,
        "fatah_text_description": 0,
        "hamas_text_username": 0,
        "fatah_text_username": 0
    }
    metadata = []

    # Check description
    for kw in hamas_keywords:
        if kw in description:
            flags["hamas_text_description"] = 1
            metadata.append({
                "source": "description",
                "text": description,
                "affiliation": "hamas"
            })
            break
    for kw in fatah_keywords:
        if kw in description:
            flags["fatah_text_description"] = 1
            metadata.append({
                "source": "description",
                "text": description,
                "affiliation": "fatah"
            })
            break

    # Check username
    for kw in hamas_keywords:
        if kw in username:
            flags["hamas_text_username"] = 1
            metadata.append({
                "source": "username",
                "text": username,
                "affiliation": "hamas"
            })
            break
    for kw in fatah_keywords:
        if kw in username:
            flags["fatah_text_username"] = 1
            metadata.append({
                "source": "username",
                "text": username,
                "affiliation": "fatah"
            })
            break

    # Check OCR on frames
    for filename in os.listdir(frames_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            text = extract_text_from_frame(os.path.join(frames_dir, filename))
            frame_metadata = {
                "source": "ocr",
                "frame": filename,
                "text": text,
                "affiliation": None
            }
            if any(kw in text for kw in hamas_keywords):
                flags["hamas_text_ocr"] = 1
                frame_metadata["affiliation"] = "hamas"
            if any(kw in text for kw in fatah_keywords):
                flags["fatah_text_ocr"] = 1
                frame_metadata["affiliation"] = "fatah"
            if frame_metadata["affiliation"] is not None:
                metadata.append(frame_metadata)
            if flags["hamas_text_ocr"] and flags["fatah_text_ocr"]:
                break  # Stop if both are found

    return flags, metadata
