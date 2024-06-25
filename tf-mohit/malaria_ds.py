import tensorflow as tf
import tensorflow_datasets as tfds
import os
from PIL import Image

# Set the number of images to download
NUM_IMAGES = 30

# Create a directory to save the images
save_dir = 'malaria_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load the malaria dataset
dataset, dataset_info = tfds.load('malaria', split='train', with_info=True, as_supervised=True)

# Define a function to save the images
def save_image(image, label, index):
    # Convert the image to a PIL Image
    image = tf.image.convert_image_dtype(image, tf.uint8)
    image = Image.fromarray(image.numpy())
    
    # Create the filename and save the image
    label_str = 'parasitized' if label == 1 else 'uninfected'
    filename = os.path.join(save_dir, f'{label_str}_{index}.png')
    image.save(filename)

# Iterate through the dataset and save the first NUM_IMAGES images
for index, (image, label) in enumerate(dataset.take(NUM_IMAGES)):
    save_image(image, label, index)
    print(f'Saved image {index + 1}/{NUM_IMAGES}')
