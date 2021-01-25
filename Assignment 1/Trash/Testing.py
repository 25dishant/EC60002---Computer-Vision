import numpy as np
# import pandas as pd


# fd = 'Data.xlsx'
# datas = pd.read_excel(fd)
# datas['X'] = datas['x'] * datas['philips_spectra']
# datas['Y'] = datas['y'] * datas['philips_spectra']
# datas['Z'] = datas['z'] * datas['philips_spectra']
# X = datas['X'].sum()
# Y = datas['Y'].sum()
# Z = datas['Z'].sum()

A = np.array([[0.49, 0.31, 0.2],
              [0.177, 0.813, 0.01],
              [0, 0.01, 0.99]]
             )
B = np.linalg.inv(A)
# XYZ = [X, Y, Z]
# rgb = np.dot(B, XYZ)
# print(rgb[0])
print(B)
