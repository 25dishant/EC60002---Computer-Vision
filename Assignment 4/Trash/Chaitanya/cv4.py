# -*- coding: utf-8 -*-
"""CV4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JqWPPirlYMPU_jjHegrRq9jfiyF27wsv
"""

from google.colab import drive
drive.mount('/gdrive')

from cv2 import imread
from cv2 import imshow
from cv2 import filter2D
import cv2
import csv
import glob
import numpy as np
import random 
import math
import skimage
import matplotlib.pyplot as plt
from skimage.viewer import ImageViewer
from google.colab.patches import cv2_imshow

from google.colab import drive
drive.mount('/content/drive')

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y+1))     
    val_ar.append(get_pixel(img, center, x, y+1))      
    val_ar.append(get_pixel(img, center, x+1, y+1))     
    val_ar.append(get_pixel(img, center, x+1, y))       
    val_ar.append(get_pixel(img, center, x+1, y-1))     
    val_ar.append(get_pixel(img, center, x, y-1))       
    val_ar.append(get_pixel(img, center, x-1, y-1))     
    val_ar.append(get_pixel(img, center, x-1, y))       
    
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val

def hist_0deg(domangle, lbp, max, dom_val, rows, cols):
    hist = np.zeros(256, dtype='uint8')
    for i in range(rows):
        for j in range(cols):
            if domangle[i, j] == 0 and dom_val[i, j] > max:
                hist[lbp[i, j]] += 1
    return hist


def hist_45deg(domangle, lbp, max, dom_val, rows, cols):
    hist = np.zeros(256, dtype='uint8')
    for i in range(rows):
        for j in range(cols):
            if domangle[i, j] == 45 and dom_val[i, j] >= max:
                hist[lbp[i, j]] += 1
    return hist


def hist_90deg(domangle, lbp, max, dom_val, rows, cols):
    hist = np.zeros(256, dtype='uint8')
    for i in range(rows):
        for j in range(cols):
            if domangle[i, j] == 90 and dom_val[i, j] >= max:
                hist[lbp[i, j]] += 1
    return hist

def hist_135deg(domangle, lbp, max, dom_val, rows, cols):
    hist = np.zeros(256, dtype='uint8')
    for i in range(rows):
        for j in range(cols):
            if domangle[i, j] == 135 and dom_val[i, j] >= max:
                hist[lbp[i, j]] += 1
    return hist

def show_output(output_list):
    output_list_len = len(output_list)
    figure = plt.figure()
    for i in range(output_list_len):
        current_dict = output_list[i]
        current_img = current_dict["img"]
        current_xlabel = current_dict["xlabel"]
        current_ylabel = current_dict["ylabel"]
        current_xtick = current_dict["xtick"]
        current_ytick = current_dict["ytick"]
        current_title = current_dict["title"]
        current_type = current_dict["type"]
        current_plot = figure.add_subplot(1, output_list_len, i+1)
        if current_type == "gray":
            current_plot.imshow(current_img, cmap = plt.get_cmap('gray'))
            current_plot.set_title(current_title)
            current_plot.set_xticks(current_xtick)
            current_plot.set_yticks(current_ytick)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)
        elif current_type == "histogram":
            current_plot.plot(current_img, color = "black")
            current_plot.set_xlim([0,260])
            current_plot.set_title(current_title)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)            
            ytick_list = [int(i) for i in current_plot.get_yticks()]
            current_plot.set_yticklabels(ytick_list,rotation = 90)

    plt.show()

def gabor_kernel(sig,lambda_x,theta):
    mu = 0.5
    l = math.ceil(6*sig)
    if(not(l%2)):
       l = l + 1;
    gabor = np.zeros((l, l), np.float32)
    m = int(l//2)
    n = int(l//2)
    for i in range(-m, m+1):
        for j in range(-n, n+1):
            x = i*math.cos(math.radians(theta)) + j*math.sin(math.radians(theta))
            y = j*math.cos(math.radians(theta)) - i*math.sin(math.radians(theta))
            g = np.exp(-(np.square(x) + np.square(mu*y))/(2*np.square(sig)))
            c= math.cos(2*np.pi*x/lambda_x)
            gabor[i+m, j+n] = g*c
    
    return gabor

for images in glob.glob("/content/drive/MyDrive/3grayimages/*.*"):
  img = imread(images,0)
 # cv2_imshow(img)
  height, width = img.shape
  img_lbp = np.zeros_like(img, dtype=np.uint8)
  kernel1 = gabor_kernel(3,5,0)
  im_filtered1 = filter2D(img,-1,kernel1)
  OP_1 = im_filtered1.astype(np.uint8)
  cv2_imshow(OP_1)
  kernel2 = gabor_kernel(3,5,45)
  im_filtered2 = filter2D(img,-1,kernel2)
  OP_2 = im_filtered2.astype(np.uint8)
  cv2_imshow(OP_2)
  kernel3 = gabor_kernel(3,5,90)
  im_filtered3 = filter2D(img,-1,kernel3)
  OP_3 = im_filtered3.astype(np.uint8)
  cv2_imshow(OP_3)
  kernel4 = gabor_kernel(3,5,135)
  im_filtered4 = filter2D(img,-1,kernel4)
  OP_4 = im_filtered4.astype(np.uint8)
  cv2_imshow(OP_4)
  dominant = np.zeros_like(img, dtype=np.float32) 
  a = np.zeros_like(img, dtype=np.float32) 
  for i in range (0,height):
      for j in range (0,width):
        dominant[i][j] = max(abs(OP_1[i][j]),abs(OP_2[i][j]),abs(OP_3[i][j]),abs(OP_4[i][j]))
        if(dominant[i][j] == abs(OP_1[i][j])):
          a[i][j] = 0;
        if(dominant[i][j] == abs(OP_2[i][j])):
          a[i][j] = 45;
        if(dominant[i][j] == abs(OP_3[i][j])):
          a[i][j] = 90;
        if(dominant[i][j] == abs(OP_4[i][j])):
          a[i][j] = 135;
  max_G= 0.1* np.max(dominant)
  for i in range(0, height):
      for j in range(0, width):
             img_lbp[i, j] = lbp_calculated_pixel(img, i, j)
  hist_LBP_0deg = hist_0deg(a, img_lbp, max_G, dominant, height, width)
  hist_LBP_45deg = hist_45deg(a, img_lbp, max_G, dominant, height, width)
  hist_LBP_90deg = hist_90deg(a, img_lbp, max_G, dominant, height, width)
  hist_LBP_135deg = hist_135deg(a, img_lbp, max_G, dominant, height, width)

  output_list1 = []
  output_list1.append({
        "img": hist_LBP_0deg,
        "xlabel": "Bins",
        "ylabel": "Number of pixels",
        "xtick": None,
        "ytick": None,
        "title": "Histogram(LBP)_45deg",
        "type": "histogram"
    })

  show_output(output_list1)
  output_list2 = []
  output_list2.append({
        "img": hist_LBP_45deg,
        "xlabel": "Bins",
        "ylabel": "Number of pixels",
        "xtick": None,
        "ytick": None,
        "title": "Histogram(LBP)_45deg",
        "type": "histogram"
    })

  show_output(output_list2)
  output_list3 = []
  output_list3.append({
        "img": hist_LBP_90deg,
        "xlabel": "Bins",
        "ylabel": "Number of pixels",
        "xtick": None,
        "ytick": None,
        "title": "Histogram(LBP)_90deg",
        "type": "histogram"
    })

  show_output(output_list3)
  output_list4 = []
  output_list4.append({
        "img": hist_LBP_135deg,
        "xlabel": "Bins",
        "ylabel": "Number of pixels",
        "xtick": None,
        "ytick": None,
        "title": "Histogram(LBP)_135deg",
        "type": "histogram"
    })

  show_output(output_list4)


  print("--------------------------------------------")

