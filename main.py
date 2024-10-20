from models.model import create_model, train_model
from models.utils import preprocess_image
import os

def main():
  
    data_dir = 'data/dataset/'
    images = []
    labels = []

    for label in os.listdir(data_dir):
        for image_file in os.listdir(os.path.join(data_dir, label)):
            image_path = os.path.join(data_dir, label, image_file)
            image = preprocess_image(image_path)
            images.append(image)
            labels.append(int(label))  # Assuming labels are integer

    images = np.array(images)
    labels = np.array(labels)

   
    model = create_model((128, 128, 3))
    train_model(model, images, labels)

if __name__ == '__main__':
    main()

