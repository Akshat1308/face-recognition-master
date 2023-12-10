import os
import cv2
import numpy as np
from imgaug import augmenters as iaa

# Define the input and output folder paths
input_folder = "/home/shristi/Desktop/face-recognition-master/augmentation/input"
output_folder = "/home/shristi/Desktop/face-recognition-master/augmentation/output"

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define the augmentation parameters
augmentation_params = iaa.Sequential([
    iaa.Rotate((-30, 30)),               # Rotate images up to 20 degrees
    iaa.Fliplr(0.5),                    # Flip horizontally with a 50% probability
    iaa.Affine(scale=(0.8, 1.2)),       # Scale images to 80-120% of their size
    iaa.Affine(shear=(-20, 20)),        # Shear images by -20 to 20 degrees
    iaa.CropAndPad(percent=(-0.2, 0.2)), # Crop and pad images
])

# Function to apply augmentation to a single image
def augment_image(image):
    return augmentation_params.augment_image(image)

# Function to add random noise to an image
def add_noise(image):
    noise = np.random.normal(0, 2, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

# Recursively walk through the input folder
for root, dirs, files in os.walk(input_folder):
    # Create a corresponding subfolder in the output folder
    relative_path = os.path.relpath(root, input_folder)
    output_subfolder = os.path.join(output_folder, relative_path)
    os.makedirs(output_subfolder, exist_ok=True)

    # Iterate through image files in the current subfolder
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            input_image_path = os.path.join(root, file)
            output_image_path = os.path.join(output_subfolder, file)

            # Read the input image
            input_image = cv2.imread(input_image_path)

            # Augment the image 10 times
            for i in range(10):
                augmented_image = augment_image(input_image)

                # Add noise to some of the augmented images (e.g., every second image)
                if i % 2 == 0:
                    augmented_image = add_noise(augmented_image)

                augmented_image_path = output_image_path.replace(
                    os.path.splitext(file)[0], f"{os.path.splitext(file)[0]}_aug_{i}"
                )
                cv2.imwrite(augmented_image_path, augmented_image)

print("Image augmentation with noise addition completed.")