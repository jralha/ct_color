#%% Libs
import glob
import cv2
import numpy as np
from tqdm.auto import tqdm

#%% Load imgs
img_array = []
files = glob.glob('ct_color\\style_results_bkp\\*.png')
nfiles = len(files)
for i in tqdm(range(nfiles),position=0):
    filename = 'ct_color\\style_results_bkp\\'+'epoch_'+str(i)+'.png'
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 

#%% Make video
out = cv2.VideoWriter('ct_color\\project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 20, size)

k = 0
for i in tqdm(range(len(img_array[0:50])),position=0):
    out.write(img_array[k])
    k=k+1
for i in tqdm(range(len(img_array[50:1000:5])),position=0):
    out.write(img_array[k])
    k=k+5
for i in tqdm(range(len(img_array[1000:-1000:1000])),position=0):
    out.write(img_array[k])
    k=k+1000
out.release()

# %%