# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 13:22:03 2019

@author: tanma
"""

import numpy as np 
import pandas as pd 
import cv2
import matplotlib.pyplot as plt
import os


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
data = pd.read_csv("/kaggle/input/pku-autonomous-driving/train.csv")

# model type, yaw, pitch, roll, x, y, z
# ALL PREDICTION STRINGS HAVE A  LENGTH WHICH IS A MULTIPLE OF 7 #

img = cv2.imread("/kaggle/input/pku-autonomous-driving/train_images/ID_2856c07db.jpg")
fig = plt.figure(figsize=(10,10))
plt.imshow(img,cmap="gray")

img_mask = cv2.imread("/kaggle/input/pku-autonomous-driving/train_masks/ID_2856c07db.jpg")
fig = plt.figure(figsize=(10,10))
plt.imshow(img_mask)

img = cv2.resize(img,(224,224))
img_mask = cv2.resize(img_mask,(224,224))
masked = cv2.bitwise_and(img,img,img_mask)

fig = plt.figure(figsize=(10,10))
plt.imshow(masked)

test_str = data[data["ImageId"]=="ID_2856c07db"]["PredictionString"].values

test_list = test_str[0].split()
test_dict = {}
test_dict["Model"] = []
test_dict["yaw"] = []
test_dict["pitch"] = []
test_dict["roll"] = []
test_dict["x"] = []
test_dict["y"] = []
test_dict["z"] = []
i=0
while(i<len(test_list)):
    
    test_dict["Model"].append(float(test_list[i]))
    test_dict["yaw"].append(float(test_list[i+1]))
    test_dict["pitch"].append(float(test_list[i+2]))
    test_dict["roll"].append(float(test_list[i+3]))
    test_dict["x"].append(float(test_list[i+4]))
    test_dict["y"].append(float(test_list[i+5]))
    test_dict["z"].append(float(test_list[i+6]))
    i = i+7
    
def str2coords(s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    '''
    Input:
        s: PredictionString (e.g. from train dataframe)
        names: array of what to extract from the string
    Output:
        list of dicts with keys from `names`
    '''
    coords = []
    
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
        if 'id' in coords[-1]:
            coords[-1]['id'] = int(coords[-1]['id'])
    return coords




PATH = "/kaggle/input/pku-autonomous-driving/"

def get_img_coords(s):
    '''
    Input is a PredictionString (e.g. from train dataframe)
    Output is two arrays:
        xs: x coordinates in the image (row)
        ys: y coordinates in the image (column)
    '''
    coords = str2coords(s)
    fx = 2304.5479
    fy = 2305.8757
    cx = 1686.2379
    cy = 1354.9849
    camera_matrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    xs = [c['x'] for c in coords]
    ys = [c['y'] for c in coords]
    zs = [c['z'] for c in coords]
    P = np.array(list(zip(xs, ys, zs))).T
    img_p = np.dot(camera_matrix, P).T
    img_p[:, 0] /= img_p[:, 2]
    img_p[:, 1] /= img_p[:, 2]
    img_xs = img_p[:, 0]
    img_ys = img_p[:, 1]
    img_zs = img_p[:, 2] # z = Distance from the camera
    return img_xs, img_ys

plt.figure(figsize=(14,14))
img = cv2.imread(PATH + 'train_images/' + data['ImageId'][2210] + '.jpg')
plt.imshow(img)
plt.scatter(*get_img_coords(data['PredictionString'][2210]), color='red', s=100)

