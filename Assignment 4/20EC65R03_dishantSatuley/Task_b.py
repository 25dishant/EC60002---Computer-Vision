# import all the required modules
import math as m
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt


# A function that returns gabor kernel of size 6*sigma^2
def GaborKernel(lamda, theta, sigma, gamma):
    kernel_size = m.ceil(6*sigma**2)
    if kernel_size % 2 == 0:
        kernel_size += 1
    p = kernel_size//2
    q = kernel_size//2
    Kernel = np.zeros((kernel_size, kernel_size))

    for x in range(-p, p+1):
        for y in range(-q, q+1):
            x0 = x*m.cos(theta) + y*m.sin(theta)
            y0 = -x*m.sin(theta) + y*m.cos(theta)
            sinusoid = m.cos((2*m.pi*x0)/lamda)
            gaussian = np.exp(-(x0**2+(gamma**2)*y0**2)/(2*(sigma**2)))
            gabor = gaussian*sinusoid
            Kernel[x+p, y+q] = gabor

    Kernel = Kernel/np.sum(Kernel)

    return Kernel


# Function that returns an image with decimal LBP at the position of each pixel of input image
def LocalBinaryPatterns(image):
    rows, columns = image.shape

    padded_image = np.pad(image, 2, mode='constant')

    for r in range(2, rows+1):
        for c in range(2, columns+1):
            Value = 0
            centre = padded_image[r, c]
            if padded_image[r-1, c-1] > centre:
                Value += 1
            padded_image[r-1, c-1] = Value
            Value = 0
            if padded_image[r-1, c] > centre:
                Value += 1
            padded_image[r-1, c] = Value
            Value = 0
            if padded_image[r-1, c+1] > centre:
                Value += 1
            padded_image[r-1, c+1] = Value
            Value = 0
            if padded_image[r, c+1] > centre:
                Value += 1
            padded_image[r, c+1] = Value
            Value = 0
            if padded_image[r+1, c+1] > centre:
                Value += 1
            padded_image[r+1, c+1] = Value
            Value = 0
            if padded_image[r+1, c] > centre:
                Value += 1
            padded_image[r+1, c] = Value
            Value = 0
            if padded_image[r+1, c] > centre:
                Value += 1
            padded_image[r+1, c] = Value
            Value = 0
            if padded_image[r+1, c-1] > centre:
                Value += 1
            padded_image[r+1, c-1] = Value
            Value = 0
            if padded_image[r, c-1] > centre:
                Value += 1
            padded_image[r, c-1] = Value

            decimal_lbp = (padded_image[r, c-1])*(2**7)+(padded_image[r+1, c-1])*(2**6)+(padded_image[r+1, c])*(2**5)+(padded_image[r+1, c+1])*(
                2**4)+(padded_image[r, c+1])*(2**3)+(padded_image[r-1, c+1])*(2**2)+(padded_image[r-1, c])*(2**1)+(padded_image[r-1, c-1])*(2**0)

            padded_image[r, c] = decimal_lbp

    return image


# These functions return the histogram at given theta orientation
def Hist_LBP_0deg(dominant_orientation, lbp, max, dominant_value):
    rows, columns = dominant_orientation.shape
    histogram = np.zeros(256, dtype='uint8')
    for r in range(rows):
        for c in range(columns):
            if dominant_orientation[r, c] == 0 and dominant_value[r, c] > max:
                histogram[lbp[r, c]] += 1
    return histogram


def Hist_LBP_45deg(dominant_orientation, lbp, max, dominant_value):
    rows, columns = dominant_orientation.shape
    histogram = np.zeros(256, dtype='uint8')
    for r in range(rows):
        for c in range(columns):
            if dominant_orientation[r, c] == 45 and dominant_value[r, c] > max:
                histogram[lbp[r, c]] += 1
    return histogram


def Hist_LBP_90deg(dominant_orientation, lbp, max, dominant_value):
    rows, columns = dominant_orientation.shape
    histogram = np.zeros(256, dtype='uint8')
    for r in range(rows):
        for c in range(columns):
            if dominant_orientation[r, c] == 90 and dominant_value[r, c] > max:
                histogram[lbp[r, c]] += 1
    return histogram


def Hist_LBP_135deg(dominant_orientation, lbp, max, dominant_value):
    rows, columns = dominant_orientation.shape
    histogram = np.zeros(256, dtype='uint8')
    for r in range(rows):
        for c in range(columns):
            if dominant_orientation[r, c] == 135 and dominant_value[r, c] > max:
                histogram[lbp[r, c]] += 1
    return histogram


if __name__ == "__main__":
    # We are given the following values of theta
    thetas = [0, 45, 90, 135]

    # Read the image one by one
    image_list = os.listdir('assignment 4 images')

    # For each image we will apply all the operations
    image_num = 0
    while (image_num < 3):
        image = cv.imread(f'assignment 4 images/{image_list[image_num]}')
        image_num += 1
        # Convert the image to grayscale as the cv.imread() function returns a BGR image
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Convert the image into a numpy array
        image = np.array(image)

        cv.imshow("Original Image", image)
        cv.waitKey(0)

        # Rows and columns of the image
        rows, columns = image.shape

        # Make an empty list list to keep output images
        output_images = []

        # For each theta, apply the gabor filter on the image and put the result in the output_images list
        for theta in thetas:
            kernel = GaborKernel(5, theta, 3, 0.5)
            image_gabor = cv.filter2D(image, -1, kernel)
            cv.imshow(f'Image{image_num} after applying Gabor', image_gabor)
            cv.waitKey(0)
            output_images.append(image_gabor)

        # Make zero matrices of the image size for the pixel labels and the dominant orientation
        pixel_label = np.zeros(image.shape)
        dominant_orientation = np.zeros(image.shape)

        # Labeling each pixel with its dominant orientation
        for r in range(rows):
            for c in range(columns):
                maximum = max(abs(output_images[0][r, c]), abs(output_images[1]
                                                               [r, c]), abs(output_images[2][r, c]), abs(output_images[3][r, c]))
                pixel_label[r, c] = maximum
                if maximum == output_images[0][r, c]:
                    dominant_orientation[r, c] = 0
                if maximum == output_images[1][r, c]:
                    dominant_orientation[r, c] = 45
                if maximum == output_images[2][r, c]:
                    dominant_orientation[r, c] = 90
                if maximum == output_images[3][r, c]:
                    dominant_orientation[r, c] = 135

        # Calculate the linear binary patterns for the image
        image_LBP = LocalBinaryPatterns(image)
        cv.imshow(f"LBP of Image{image_num}", image_LBP)
        cv.waitKey(0)

        # Calculate the histogram for the image at asked orientation
        hist_0 = Hist_LBP_0deg(dominant_orientation, image_LBP,
                               0.1*np.max(pixel_label), pixel_label)
        hist_1 = Hist_LBP_45deg(dominant_orientation, image_LBP,
                                0.1*np.max(pixel_label), pixel_label)
        hist_2 = Hist_LBP_90deg(dominant_orientation, image_LBP,
                                0.1*np.max(pixel_label), pixel_label)
        hist_3 = Hist_LBP_135deg(dominant_orientation, image_LBP,
                                 0.1*np.max(pixel_label), pixel_label)

        # Plot all the histograms
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(hist_0)
        axs[0, 0].set_title('0 degrees')
        axs[0, 1].plot(hist_1)
        axs[0, 1].set_title('45 degrees')
        axs[1, 0].plot(hist_2)
        axs[1, 0].set_title('90 degrees')
        axs[1, 1].plot(hist_3)
        axs[1, 1].set_title('135 degrees')
        plt.tight_layout()
        plt.show()

        cv.waitKey(0)
