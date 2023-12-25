import os
import numpy as np
import skimage.io as ski_io
import matplotlib.pyplot as plt
from skimage.util import random_noise
from scipy.ndimage import median_filter

class FiltersImage:
    def __init__(self, image_name, box_size=3, kernel_size=3):
        self.image = self.read_image(image_name)
        self.noisy_image = None
        self.filtered_image = None

        self.box_size = box_size
        self.kernel_size = kernel_size

    def read_image(self, image_name):
        image_path = os.path.abspath(f'{image_name}.png')
        return ski_io.imread(image_path)

    def noisy_filter(self, mode='s&p', intensity=0.05):
        if mode == 's&p':
            self.noisy_image = random_noise(self.image, mode='s&p', amount=intensity)
        else:
            self.noisy_image = random_noise(self.image, mode='gaussian', var=intensity)

    def box_filter(self):
        H, W, channels = self.image.shape
        box_half = self.box_size // 2

        new_img = np.zeros_like(self.image)
        for c in range(channels):
            for i in range(box_half, H - box_half):
                for j in range(box_half, W - box_half):
                    new_img[i, j, c] = np.mean(
                        self.image[i - box_half:i + box_half + 1, j - box_half:j + box_half + 1, c])

        return new_img

    def median_filter(self):
        H, W, channels = self.image.shape
        kernel_half = self.kernel_size // 2

        new_img = np.zeros_like(self.image)

        for c in range(channels):
            for i in range(kernel_half, H - kernel_half):
                for j in range(kernel_half, W - kernel_half):
                    new_img[i, j, c] = np.median(
                        self.image[i - kernel_half:i + kernel_half + 1, j - kernel_half:j + kernel_half + 1, c])

        return new_img

    def weighted_median(self):
        new_img = np.zeros_like(self.image)
        for c in range(self.image.shape[2]):
            new_img[:, :, c] = median_filter(self.image[:, :, c], size=self.kernel_size)

        return new_img

    def build_plot(self):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(self.image, vmin=0, vmax=1)
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        axs[1].imshow(self.noisy_image, vmin=0, vmax=1)
        axs[1].set_title('Noised Image')
        axs[1].axis('off')

        axs[2].imshow(self.filtered_image, vmin=0, vmax=1)
        axs[2].set_title('Filtered Image')
        axs[2].axis('off')

        plt.show()

# Use your image names here
image_names = ['img_1', 'img_2', 'img_3']

for name in image_names:
    capy_image = FiltersImage(name)
    capy_image.noisy_filter(mode='s&p', intensity=0.1)
    capy_image.filtered_image = capy_image.box_filter()  # Change to the desired filter function
    capy_image.build_plot()
