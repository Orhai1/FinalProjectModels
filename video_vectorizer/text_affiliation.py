def extract_text_from_frame(path):
    # Call EasyOCR / pytesseract here
    return "text"

hamas_keywords = {"hamas", "qassam", "yahya"}
fatah_keywords = {"fatah", "arafat", "dahlan"}
all_keywords = hamas_keywords | fatah_keywords

def check_text_sources(frames_dir, description, username):
    flags = {
        "affiliation_from_ocr": 0,
        "affiliation_from_description": 0,
        "affiliation_from_username": 0
    }

    if any(k in description.lower() for k in all_keywords):
        flags["affiliation_from_description"] = 1

    if any(k in username.lower() for k in all_keywords):
        flags["affiliation_from_username"] = 1

    for filename in os.listdir(frames_dir):
        text = extract_text_from_frame(os.path.join(frames_dir, filename))
        if any(k in text.lower() for k in all_keywords):
            flags["affiliation_from_ocr"] = 1
            break

    return flags
