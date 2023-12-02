import cv2
import numpy as np

def apply_filter_and_show(image, filter_matrix, output_filename):
    result = cv2.filter2D(image, -1, filter_matrix)
    cv2.imshow('Filtered Image', result)
    cv2.imwrite(output_filename, result)

def main():
    # Завантаження зображення
    image = cv2.imread('img_1.png')

    # Отримання висоти та ширини зображення
    height, width = image.shape[:2]

    # Зсув зображення
    shift_filter = np.float32([[1, 0, 10], [0, 1, 20], [0, 0, 1]])
    apply_filter_and_show(image, shift_filter[:2, :], "img_shifted.png")

    # Інверсія
    inversion_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    apply_filter_and_show(image, inversion_filter, "img_inversion.png")

    # Згладжування по Гауссу
    gauss_filter = np.array([[1, 2, 1, 2, 4, 2, 1, 2, 1, 2, 1],
                             [2, 4, 2, 1, 2, 1, 2, 4, 2, 1, 2],
                             [1, 2, 1, 2, 4, 2, 1, 2, 1, 2, 1],
                             [2, 4, 2, 1, 2, 1, 2, 4, 2, 1, 2],
                             [1, 2, 1, 2, 4, 2, 1, 2, 1, 2, 1],
                             [2, 4, 2, 1, 2, 1, 2, 4, 2, 1, 2],
                             [1, 2, 1, 2, 4, 2, 1, 2, 1, 2, 1],
                             [2, 4, 2, 1, 2, 1, 2, 4, 2, 1, 2],
                             [1, 2, 1, 2, 4, 2, 1, 2, 1, 2, 1],
                             [2, 4, 2, 1, 2, 1, 2, 4, 2, 1, 2],
                             [1, 2, 1, 2, 4, 2, 1, 2, 1, 2, 1]], dtype=np.float32) * (1/16)
    apply_filter_and_show(image, gauss_filter, "img_gauss.png")

    # Розмиття "рух по дiагоналi"
    blur_filter = np.array([[1, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 1]], dtype=np.float32) * (1/3)
    apply_filter_and_show(image, blur_filter, "img_blur.png")

    # Підвищення різкості
    sharp_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    apply_filter_and_show(image, sharp_filter, "img_sharp.png")

    # Фільтр Собеля
    sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    apply_filter_and_show(image, sobel_filter, "img_sobel.png")

    # Фільтр границь
    border_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    apply_filter_and_show(image, border_filter, "img_border.png")

    # Ваш цікавий фільтр
    interesting_filter = np.array([[0, 1, 0], [1, -2, 1], [0, 1, 0]], dtype=np.float32)
    apply_filter_and_show(image, interesting_filter, "img_interesting.png")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

