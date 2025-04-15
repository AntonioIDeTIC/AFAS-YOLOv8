#!/usr/bin/python3
import numpy as np

class FireEffectGenerator:
    def __init__(self):
        pass

    # Function to create a Gaussian distribution with specified mean and standard deviation
    def generate_gaussian(self, mean, std_dev, size):
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        x_center, y_center = size // 2, size // 2
        gaussian = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * std_dev ** 2))
        return mean * gaussian

    def generate_fire(self, image, size, fire_source_intensity, fire_source_size):
        # Create the initial "fire source" with high intensity
        if image.dtype == 'uint16':
            fire_source = self.generate_gaussian(fire_source_intensity, fire_source_size, size)

            # Convert the fire_source array to uint16 and add it to the image
            fire_source = fire_source.astype(np.uint16)

            fire = np.minimum(image + fire_source, 65535)  # Ensure pixel values don't exceed 16-bit limit

        else:
            fire_source = self.generate_gaussian(fire_source_intensity, fire_source_size, size)

            # Convert the fire_source array to uint8 and add it to the image
            fire_source = fire_source.astype(np.uint8)

            fire = np.minimum(image + fire_source, 255)
        return fire

    def apply_fire(self, image, x, y, size, fire_source_intensity, fire_source_size):
        # Create the fire mask with the same size as the original image
        fire_mask = self.generate_fire(np.zeros(image.shape, dtype=np.uint16), size, fire_source_intensity, fire_source_size)

        # Ensure that the fire region fits within the original image
        min_x = max(0, x - size // 2)
        max_x = min(image.shape[1], x + size // 2)
        min_y = max(0, y - size // 2)
        max_y = min(image.shape[0], y + size // 2)

        # Overlay the fire mask on the original image within the specified region
        image[min_y:max_y, min_x:max_x] += fire_mask[min_y - y + size // 2:max_y - y + size // 2,
                                                     min_x - x + size // 2:max_x - x + size // 2]

        return image
