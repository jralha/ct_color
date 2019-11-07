#%%
from tensorflow.keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
import nibabel as nib
import numpy as np
import glob
import os
import random
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000 

#Fun
def load_img(path):
    img = Image.open(path)
    img = np.array(img, dtype=float)
    if img.shape[2] == 4:
        img = img[:,:,:-1]
    return img

def img_to_lab(image):
    grays = rgb2lab(1.0/255*image)[:,:,0]
    ab = rgb2lab(1.0/255*image)[:,:,1:]
    ab /= 128
    height=image.shape[0]
    width=image.shape[1]
    grays = grays.reshape(1, height, width, 1)
    ab = ab.reshape(1, height, width, 2)

    return grays, ab

# %% Load photos
foto_folder = 'ct_color\\DADOS_Joao\\ARQUIVOS_HDI'
Xs=[]
Ys=[]
for file in (glob.glob(foto_folder + "/**/*.png",recursive=True)):
    
    foto = load_img(file)
    X,Y = img_to_lab(foto)
    Xs.append(X)
    Ys.append(Y)



# %%
