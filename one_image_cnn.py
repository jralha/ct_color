#%%
from tensorflow.keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
from PIL import Image
import numpy as np
import os
import random
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img

#%%
def load_crop(path,shape):
    img = Image.open(path)
    img = img.crop(shape)
    img = np.array(img, dtype=float)
    if img.shape[2] == 4:
        img = img[:,:,:-1]
    return img

foto = load_crop('ct_color\\output0\\reg_foto.png',(0,0,96,800))
ct = load_crop('ct_color\\dados0\\Tomografia_Cx6_T2.PNG',(0,0,96,800))

height=foto.shape[0]
width=foto.shape[1]

image=foto
X = rgb2lab(1.0/255*image)[:,:,0]
Y = rgb2lab(1.0/255*image)[:,:,1:]
Y /= 128
X = X.reshape(1, height, width, 1)
Y = Y.reshape(1, height, width, 2)

image=ct
ctX = rgb2lab(1.0/255*image)[:,:,0]
ctY = rgb2lab(1.0/255*image)[:,:,1:]
ctY /= 128
ctX = ctX.reshape(1, height, width, 1)
ctY = ctY.reshape(1, height, width, 2)

#%%# Building the neural network
model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))

# %%# Finish model
model.compile(optimizer='rmsprop', loss='mse')
model.fit(x=X,y=Y,batch_size=1,epochs=1,verbose=0)

#%%
print(model.evaluate(X, Y, batch_size=1))


#%%
output = model.predict(ctX)
output *= 128
# Output colorizations
cur = np.zeros((height, width, 3))
cur[:,:,0] = ctX[0][:,:,0]
cur[:,:,1:] = output[0]
plt.imshow(lab2rgb(cur))


# %%
