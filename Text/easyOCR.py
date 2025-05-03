import easyocr
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Initialize the EasyOCR reader with Arabic and English
reader = easyocr.Reader(['ar','en'], gpu=False)  # Set gpu=True if you have a GPU available

def detect_text(image_path, min_confidence=0.2):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Run EasyOCR on the image to detect text
    results = reader.readtext(image)

    # Loop over detected text and print results
    for bbox, text, confidence in results:
        if confidence >= min_confidence:  # Only print entries above the confidence threshold
            print(f"Detected Text: {text} (Confidence: {confidence:.2f})")

    # Create a copy to draw bounding boxes on
    output_image = image.copy()

    # Draw bounding boxes around the detected text
    for bbox, text, confidence in results:
        if confidence >= min_confidence:
            # Convert each point in the bounding box to integer coordinates
            pts = [tuple(map(int, point)) for point in bbox]
            cv2.polylines(output_image, [np.array(pts)], True, (0, 255, 0), 2)
            # Put the detected text above the bounding box
            cv2.putText(output_image, text, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Convert BGR to RGB for displaying with matplotlib
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    # Display the image with detections
    plt.figure(figsize=(12, 8))
    plt.imshow(output_image)
    plt.title("EasyOCR Detection")
    plt.axis("off")
    plt.show()


# Example usage
image_path = '../Symbols/video_frames/7314560502917516562_frames/7314560502917516562_frame_0002.jpg'
detect_text(image_path)

