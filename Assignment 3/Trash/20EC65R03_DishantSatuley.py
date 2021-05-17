# from google.colab.patches import cv2_imshow
# import cv2 as cv
# import math as m
# import numpy as np
# from google.colab import drive
# drive.mount('/gdrive')


sigma1 = float(input("sigma1: "))
sigma2 = float(input("sigma2: "))

# def Gaussian(sigma, mux, muy):
#     kernel_size = m.ceil(6*sigma2)
#     if kernel_size % 2 == 0:
#         kernel_size = kernel_size + 1
#     Gaussian_Kernel = np.zeros((kernel_size, kernel_size))
#     p = kernel_size//2
#     q = kernel_size//2

#     for x in range(-p, p+1):
#         for y in range(-q, q+1):
#             Gaussian = (1/(2*np.pi*sigma**2)) * \
#                 np.exp(-(((x-mux)**2)+(y-muy)**2)/2*sigma**2)

#             Gaussian_Kernel[x+1, y+1] = Gaussian

#     Gaussian_Kernel = Gaussian_Kernel/np.sum(Gaussian_Kernel)

#     return Gaussian_Kernel


# image1 = cv.imread("My Drive/assignment 3 images/Balls.tif")
# B = image1[:, :, 0]
# G = image1[:, :, 1]
# R = image1[:, :, 2]
# kernel1 = Gaussian(sigma1, 0, 0)
# kernel2 = Gaussian(sigma2, 0, 0)
# kernel = kernel1 - kernel2
# B = cv.filter2D(B, -1, kernel)
# G = cv.filter2D(G, -1, kernel)
# R = cv.filter2D(R, -1, kernel)
# image1 = abs(B)+abs(G)+abs(R)
# cv2_imshow(image1)

print(sigma1, sigma2)
