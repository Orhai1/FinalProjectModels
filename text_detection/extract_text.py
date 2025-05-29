import os
import re
import easyocr
import numpy as np
import cv2

# Initialize the OCR reader (add 'en' if you also expect English)
reader = easyocr.Reader(['ar'], gpu=False)


def normalize_arabic(text):
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'[ًٌٍَُِّْ]', '', text)  # Remove diacritics
    return text


def tokenize(text):
    return text.strip().split()


def detect_text_in_image(image_path, min_confidence=0.2):
    image = cv2.imread(image_path)
    results = reader.readtext(image)

    detected_words = []
    for _, text, conf in results:
        if conf >= min_confidence:
            detected_words.append(text)
    return detected_words


def generate_word_bank_from_images(folder_path):
    all_words = set()

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")
            detected_texts = detect_text_in_image(image_path)
            for text in detected_texts:
                normalized = normalize_arabic(text)
                tokens = tokenize(normalized)
                all_words.update(tokens)

    return all_words


# Example usage:
if __name__ == "__main__":
    image_folder = '../symbols_detection/video_frames/7314560502917516562_frames'  # replace with your folder path
    word_bank = generate_word_bank_from_images(image_folder)

    print("\n All words in frames:")
    for word in sorted(word_bank):
        print(word)

    # Optional: save to file
    # with open('word_bank.txt', 'w', encoding='utf-8') as f:
    #     for word in sorted(word_bank):
    #         f.write(word + '\n')
