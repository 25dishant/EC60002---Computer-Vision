from google.colab.patches import cv2_imshow
import math
import numpy as np
from cv2 import imread
from google.colab import drive
drive.mount('/gdrive')
%cd / gdrive


Image = imread("My Drive/CV_asgnmnt2_Images/10.tiff")
cv2_imshow(Image)
R = Image[:, :, 0]/255
G = Image[:, :, 1]/255
B = Image[:, :, 2]/255

# Theta = (180/math.pi)*(np.arccos((0.5*((R-G)+(R-B)))/(((R-G)**2+((R-B)*(G-B)))**0.5)))
x = (180/math.pi)
y = np.arccos((0.5*((R-G)+(R-B))
               z=((R-G)**2+((R-B)*(G-B)))**0.5))
Theta = x*(y/z)
if B.all() <= G.all():
    H = Theta
else:
    H = 360 - Theta
H = H / 360

I = (R+G+B)/3
S = 1 - (min(R.all(), G.all(), B.all())/I)

HSI = np.zeros([len(Image), len(Image[0]), 3])
HSI[:, :, 0] = H*255
HSI[:, :, 1] = S*255
HSI[:, :, 2] = I*255
cv2_imshow(HSI)
# len(HSI)
