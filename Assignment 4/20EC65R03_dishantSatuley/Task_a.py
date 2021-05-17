import numpy as np
import math as m
import cv2 as cv


# A function that returns gabor kernel
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

    return Kernel


# As theta and gamma are fixed we assign the given values to them
theta = 0
gamma = 0.5

# We'll try different values of lambda and sigma in order to observe the low pass nature of the gabor filter
num = 1
while True:
    # As size of the kernel is sigma dependent, I haven't considered very large values of sigma cuz the kernel window runs of the screen
    lamda = float(input("lambda = "))
    sigma = float(input("sigma = "))
    print("\n")

    kernel1 = GaborKernel(lamda, theta, sigma, gamma)

    cv.namedWindow(
        f'Image{num} with lambda = {lamda} and sigma = {sigma}', cv.WINDOW_NORMAL)
    cv.imshow(f'Image{num} with lambda = {lamda} and sigma = {sigma}', kernel1)

    num += 1
    cv.waitKey(0)


# Keeping the lambda constant when we increase the value of sigma, the number of stripes in the gobor filter increases and we run away from the low pass characterstics of the gabor filter.


# Keeping the sigma constant, when we increase the value of lambda, spread of the gabor filter increases and we move toward the low pass nature of the of gabor filter.
