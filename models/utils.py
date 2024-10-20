import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))  
    image = np.array(image) / 255.0  # Normalize
    return image
