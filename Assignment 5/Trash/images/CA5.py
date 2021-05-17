import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

#to add zero padding
def padding(image,rows,cols):
    img = np.zeros(shape = (rows+2,cols+2), dtype = 'uint8')
    img[1:rows+1,1:cols+1] = image
    return img

#median filter
def OS1(image, r, c):
    img = np.zeros(shape = (r,c), dtype = 'uint8')
    for x in range(1,r+1):
        for y in range(1,c+1):
            a = np.zeros(shape = (3,3))
            a = image[x-1:x+2,y-1:y+2]
            b = a.flatten()
            b = np.sort(b)
            img[x-1,y-1] = b[4]
    return img

#midpoint filter
def OS2(image, r, c):
    img = np.zeros(shape = (r,c), dtype = 'uint8')
    for x in range(1,r+1):
        for y in range(1,c+1):
            a = np.zeros(shape = (3,3))
            a = image[x-1:x+2,y-1:y+2]
            b = a.flatten()
            b = np.sort(b)
            value = round((int(b[0]) + int(b[8]))/2.0)
            img[x-1,y-1] = value
    return img

def sigmaxy(x,y,_x,_y):
    dx = x-_x
    dy = y-_y
    S = np.mean(dx*dy)
    return S

def SSIM(image,original,r,c,size):
    SSIM_sum = 0
    for i in range(0,r,size):
        for j in range(0,c,size):
            x = np.zeros(shape = (size,size), dtype = 'uint8')
            x = image[i:i+size+1,j:j+size+1]
            y = np.zeros(shape = (size,size), dtype = 'uint8')
            y = original[i:i+size+1,j:j+size+1]
            _x = x.flatten()
            _y = y.flatten()
            ux = np.mean(_x)
            uy = np.mean(_y)
            sx = np.std(_x)
            sy = np.std(_y)
            sxy = sigmaxy(_x,_y,ux,uy)
            SSIM_Num = 2*ux*uy*2*sxy
            SSIM_Den = ((ux**2 + uy**2)*(sx**2 + sy**2)) + 0.001
            ssim = SSIM_Num/SSIM_Den
            SSIM_sum += ssim
    return SSIM_sum/256

filelist = ['1','2','3']
for i in filelist:
    original = i + '.tiff'
    blurred = i + 'B.bmp'
    gaussian = i + 'G.bmp'
    laplacian = i + 'L.bmp'
    image_original = cv2.imread(original, cv2.IMREAD_UNCHANGED)
    image_blurred = cv2.imread(blurred, cv2.IMREAD_UNCHANGED)
    image_gaussian = cv2.imread(gaussian, cv2.IMREAD_UNCHANGED)
    image_laplacian = cv2.imread(laplacian, cv2.IMREAD_UNCHANGED)
    rows, cols = image_original.shape
    
    
    #applying OS1 to the degraded images
    padded_blur = padding(image_blurred, rows, cols)
    blurred_os1 = OS1(padded_blur,rows,cols)
    cv2.imshow("actual",image_blurred)
    cv2.imshow("after applying OS1",blurred_os1)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

    padded_gaus = padding(image_gaussian, rows, cols)
    gaussian_os1 = OS1(padded_gaus,rows,cols)
    cv2.imshow("actual",image_gaussian)
    cv2.imshow("after applying OS1",gaussian_os1)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

    padded_lap = padding(image_laplacian, rows, cols)
    laplacian_os1 = OS1(padded_lap,rows,cols)
    cv2.imshow("actual",image_laplacian)
    cv2.imshow("after applying OS1",laplacian_os1)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()     
    
    
    #applying OS2 to the degraded images
    blurred_os2 = OS2(padded_blur,rows,cols)
    cv2.imshow("actual",image_blurred)
    cv2.imshow("after applying OS2",blurred_os2)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

    gaussian_os2 = OS2(padded_gaus,rows,cols)
    cv2.imshow("actual",image_gaussian)
    cv2.imshow("after applying OS2",gaussian_os2)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

    laplacian_os2 = OS2(padded_lap,rows,cols)
    cv2.imshow("actual",image_laplacian)
    cv2.imshow("after applying OS2",laplacian_os2)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()


    #calculating SSIM
    #we will consider 256 non overlapping regions
    print("For image "+str(i)+" :")
    size = int(math.sqrt((rows*cols)/256))
    
    SSIM_orig = SSIM(image_original,image_original,rows,cols,size)
    SSIM_blurred = SSIM(image_blurred,image_original,rows,cols,size)
    SSIM_b_enhanced1 = SSIM(blurred_os1,image_original,rows,cols,size)
    SSIM_b_enhanced2 = SSIM(blurred_os2,image_original,rows,cols,size)
    
    SSIM_gaussian = SSIM(image_gaussian,image_original,rows,cols,size)
    SSIM_g_enhanced1 = SSIM(gaussian_os1,image_original,rows,cols,size)
    SSIM_g_enhanced2 = SSIM(gaussian_os2,image_original,rows,cols,size)
    
    SSIM_laplacian = SSIM(image_laplacian,image_original,rows,cols,size)
    SSIM_l_enhanced1 = SSIM(laplacian_os1,image_original,rows,cols,size)
    SSIM_l_enhanced2 = SSIM(laplacian_os2,image_original,rows,cols,size)
    
    print(SSIM_orig)
    
    print(SSIM_blurred,SSIM_gaussian,SSIM_laplacian)
    print(SSIM_b_enhanced1,SSIM_g_enhanced1,SSIM_l_enhanced1)
    print(SSIM_b_enhanced2,SSIM_g_enhanced2,SSIM_l_enhanced2)
    
    
    #showing images before applying filters
    """fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(image_original, cmap = 'gray')
    axs[0, 0].set_title('Original')
    
    axs[0, 1].imshow(image_blurred, cmap = 'gray')
    axs[0, 1].set_title('Blurred')
    
    axs[1, 0].imshow(image_gaussian, cmap = 'gray')
    axs[1, 0].set_title('Gaussian')
    
    axs[1, 1].imshow(image_laplacian, cmap = 'gray')
    axs[1, 1].set_title('Laplacian')
    plt.tight_layout()
    plt.show()"""