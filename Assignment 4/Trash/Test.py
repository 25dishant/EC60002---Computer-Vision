import math as m
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt


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

    # image = np.delete(padded_image, [0, 1, rows+2, rows+3], axis=0)
    # image = np.delete(image, [0, 1, columns+2, columns+3], axis=1)

    return image


def Hist_LBP_0deg(dominant_orientation, lbp, max, dominant_value):
    rows, columns = dominant_orientation.shape
    histogram = np.zeros(512, dtype='uint8')
    for r in range(rows):
        for c in range(columns):
            if dominant_orientation[r, c] == 0 and dominant_value[r, c] > max:
                histogram[lbp[r, c]] += 1
    return histogram


def Hist_LBP_45deg(dominant_orientation, lbp, max, dominant_value):
    rows, columns = dominant_orientation.shape
    histogram = np.zeros(512, dtype='uint8')
    for r in range(rows):
        for c in range(columns):
            if dominant_orientation[r, c] == 45 and dominant_value[r, c] > max:
                histogram[lbp[r, c]] += 1
    return histogram


def Hist_LBP_90deg(dominant_orientation, lbp, max, dominant_value):
    rows, columns = dominant_orientation.shape
    histogram = np.zeros(512, dtype='uint8')
    for r in range(rows):
        for c in range(columns):
            if dominant_orientation[r, c] == 90 and dominant_value[r, c] > max:
                histogram[lbp[r, c]] += 1
    return histogram


def Hist_LBP_135deg(dominant_orientation, lbp, max, dominant_value):
    rows, columns = dominant_orientation.shape
    histogram = np.zeros(512, dtype='uint8')
    for r in range(rows):
        for c in range(columns):
            if dominant_orientation[r, c] == 90 and dominant_value[r, c] > max:
                histogram[lbp[r, c]] += 1
    return histogram


# print(kernel)
# print(np.sum(kernel))
thetas = [0, 45, 90, 135]  # given

image_list = os.listdir('assignment 4 images')
image = cv.imread(f'assignment 4 images/{image_list[0]}')

image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

image = np.array(image)

rows, columns = image.shape

# print(rows, columns)

output_images = []
for theta in thetas:
    kernel = GaborKernel(5, theta, 3, 0.5)
    image_gabor = cv.filter2D(image, -1, kernel)
    cv.imshow(f'Image', image_gabor)
    cv.waitKey(0)
    output_images.append(image_gabor)

pixel_label = np.zeros(image.shape)
dominant_orientation = np.zeros(image.shape)

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


# print(image)

# print(pixel_label)
# print(dominant_orientation)

image_LBP = LocalBinaryPatterns(image)
# print(image_LBP)


hist_0 = Hist_LBP_0deg(dominant_orientation, image_LBP,
                       0.1*np.max(pixel_label), pixel_label)
hist_1 = Hist_LBP_45deg(dominant_orientation, image_LBP,
                        0.1*np.max(pixel_label), pixel_label)
hist_2 = Hist_LBP_90deg(dominant_orientation, image_LBP,
                        0.1*np.max(pixel_label), pixel_label)
hist_3 = Hist_LBP_135deg(dominant_orientation, image_LBP,
                         0.1*np.max(pixel_label), pixel_label)


print(hist_0)

# fig = plt.figure(figsize=(50, 50))
# plot1 = fig.add_subplot(221)
# plot2 = fig.add_subplot(222)
# plot3 = fig.add_subplot(223)
# plot4 = fig.add_subplot(224)

# plot1.scatter(hist_0)

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


# plt.show()

# cv.imshow("Image_LBP", image_LBP)


# output_images[0]


# cv.imshow('Image2', image)

# print(len(output_images))

# cv.imshow('Image', image)

cv.waitKey(0)


# cv.imshow('Image', image_gabor)

# cv.waitKey(0)
