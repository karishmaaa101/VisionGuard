import pytesseract
import cv2

def read_label(image_path):
    image = cv2.imread(image_path)
    text = pytesseract.image_to_string(image)
    return text
