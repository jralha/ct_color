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
def load_crop(path):
    img = Image.open(path)
    img = np.array(img, dtype=float)
    img = img[:,:,0:3]
    imgx = img.shape[0]
    imgy = img.shape[1]
    img_channels = img.shape[2]
    img = img.reshape((1,imgx,imgy,img_channels))
    return img

foto = load_crop('ct_color\\output0\\reg_foto.png')
ct = load_crop('ct_color\\dados0\\Tomografia_Cx6_T2.PNG')

foto = foto/255
ct = ct/255
#%%# Building the neural network
model = Sequential()
model.add(InputLayer(input_shape=(None, None, 3)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))

# %%# Finish model
model.compile(optimizer='rmsprop', loss='mse')
model.fit(x=ct,y=foto,batch_size=1,epochs=100,verbose=0)
print(model.evaluate(ct, foto, batch_size=1))


#%%
pred = model.predict(ct)
output = pred[0]
output[output < 0] = 0
output = output*255
output = output.astype(int)
plt.figure(figsize=[10,20])
plt.imshow(output)


# %%
