import cv2
import numpy as np

# Зчитуємо зображення з файлу
image = cv2.imread('img_1.png', 0)  # Зчитуємо як чорно-біле

# 1. Erosion (ерозія)
kernel = np.ones((5, 5), np.uint8)
erosion = np.zeros_like(image)
for i in range(image.shape[0] - 4):
    for j in range(image.shape[1] - 4):
        erosion[i, j] = np.min(image[i:i+5, j:j+5])

# 2. Dilation (нарощування)
dilation = np.zeros_like(image)
for i in range(image.shape[0] - 4):
    for j in range(image.shape[1] - 4):
        dilation[i, j] = np.max(image[i:i+5, j:j+5])

# 3. Opening (розмикання)
opening = np.zeros_like(image)
for i in range(image.shape[0] - 4):
    for j in range(image.shape[1] - 4):
        temp = np.min(image[i:i+5, j:j+5])
        opening[i, j] = np.max(image[i:i+5, j:j+5])

# 4. Closing (замикання)
closing = np.zeros_like(image)
for i in range(image.shape[0] - 4):
    for j in range(image.shape[1] - 4):
        temp = np.max(image[i:i+5, j:j+5])
        closing[i, j] = np.min(image[i:i+5, j:j+5])

# 5. Границі
edges = cv2.Canny(image, 100, 200)

# Зберігаємо результати в файл
cv2.imwrite('erosion_img.png', erosion)
cv2.imwrite('dilation_img.png', dilation)
cv2.imwrite('opening_img.png', opening)
cv2.imwrite('closing_img.png', closing)
cv2.imwrite('edges_img.png', edges)
