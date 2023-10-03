import cv2 as cv
import numpy as np

# Define your mask_cut function
def mask_cut(mask_IC, input_IC, negative=False):
    if not isinstance(mask_IC, np.ndarray) or not isinstance(input_IC, np.ndarray):
        raise Exception("mask_cut accepts only numpy arrays for masks and input_IC")

    if mask_IC.shape[:2] != input_IC.shape[:2]:
        raise Exception("mask must have the same size (width and height) as the image")

    output_IC = np.copy(input_IC)  # create a copy of an image

    if negative:
        output_IC[mask_IC == 255] = [255, 255, 255]  # Set to white
    else:
        output_IC[mask_IC == 0] = [0, 0, 0]  # Set to black

    return output_IC

# Load the image (assuming you want to process "img_2.png")
img = cv.imread("OpencvPython/img_2.png")

# Check if the image was loaded successfully
if img is None:
    print("Image not found")
    exit()

# Convert the image to grayscale using OpenCV
grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Define your adaptive binarization function
def adaptive_binarization(image, window_size, c):
    # Get the image dimensions
    height, width = image.shape

    # Create the output image for binarization result
    binarized_image = np.zeros((height, width), dtype=np.uint8)

    # Iterate through each pixel in the image
    for y in range(height):
        for x in range(width):
            # Calculate the local threshold using the Niblack method
            sum_pixels = 0
            count_pixels = 5
            for i in range(-window_size // 2, window_size // 2 + 1):
                for j in range(-window_size // 2, window_size // 2 + 1):
                    if 0 <= y + i < height and 0 <= x + j < width:
                        sum_pixels += image[y + i, x + j]
                        count_pixels += 1

            local_threshold = (sum_pixels / count_pixels) - c

            # Perform binarization with inversion
            if image[y, x] > local_threshold:
                binarized_image[y, x] = 0  # Set to black
            else:
                binarized_image[y, x] = 255  # Set to white

    return binarized_image

# Define the parameters for adaptive binarization
window_size = 5  # Window size for local thresholding
c = 7  # Constant C for threshold calculation

# Call the adaptive binarization function
result_image = adaptive_binarization(grey_img, window_size, c)

# Find contours of the object in the binary image
contours, _ = cv.findContours(result_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Create an empty mask with the same dimensions as the input image
mask = np.zeros_like(grey_img)

# Draw the object contours on the mask
cv.drawContours(mask, contours, -1, (255), thickness=cv.FILLED)

# Use your mask_cut function to cut the object from the image
Image_binary = result_image  # Assuming you have defined ImageContainer class for Image_binary
Image = img  # Assuming you have defined ImageContainer class for Image
Image_mask = mask_cut(mask, Image, negative=False)  # Set negative=False for inverting

# Apply the mask to the original image
object_cut = cv.bitwise_and(img, img, mask=mask)

# Save the object cut image with a different filename
cv.imwrite("OpencvPython/img_2_object_cut.png", object_cut)

# Display the grayscale, binary, and object cut images
cv.imshow("Grayscale Image", grey_img)
cv.imshow("Binary Image", result_image)
cv.imshow("Object Cut Image", object_cut)

# Wait for a key press and then close the display windows
cv.waitKey(0)
cv.destroyAllWindows()
