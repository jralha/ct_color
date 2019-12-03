"""CNN based on all images on split folder"""
#%%Libraries

#Standard libraries
import glob
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#General
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt

#Keras
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, InputLayer
from tensorflow.keras.models import Sequential
tf.autograph.set_verbosity(0)


#Handling images
from skimage.color import rgb2lab, lab2rgb
import cv2

#Fun
def load_img(path):
    imageArray = cv2.imread(path)[:,:,:3]
    return imageArray

def img_to_lab(image):
    grays = rgb2lab(1.0/255*image)[:,:,0]
    ab = rgb2lab(1.0/255*image)[:,:,1:]
    ab /= 128
    height=image.shape[0]
    width=image.shape[1]
    grays = grays.reshape(1, height, width, 1)
    ab = ab.reshape(1, height, width, 2)

    return grays, ab

foto_folder = 'ct_color\\split'
all_photos = glob.glob(foto_folder + "/**/*.png",recursive=True)[0::10]

#%% Convert to Lab
print('Loading training images...')
test_img = load_img(all_photos[0])
imgX = test_img.shape[0]
imgY = test_img.shape[1]
X = np.zeros((len(all_photos),imgX,imgY,1))
Y = np.zeros((len(all_photos),imgX,imgY,2))

for pos,foto in tqdm(enumerate(all_photos),position=0):
    img = load_img(foto)
    if img.shape[0] == imgX and img.shape[1] == imgY:
        X[pos], Y[pos] = img_to_lab(img)
    else:
        img = cv2.resize(img, dsize=(imgX, imgY), interpolation=cv2.INTER_CUBIC)
        X[pos], Y[pos] = img_to_lab(img)

#%%# Building the neural network
print('Compiling Model...')
model = Sequential()
model.add(InputLayer(input_shape=(imgX, imgY, 1)))
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

#%%
model.compile(optimizer='rmsprop', loss='mse')
# %%# Finish model
print('Training model...')
model.fit(x=X,y=Y,batch_size=1,epochs=100,verbose=1)
     


# %%
print('Making prediction...')
ct_path = 'ct_color\\dados0\\Tomografia_Cx6_T2.PNG'
ct = load_img(ct_path)
ctH = ct.shape[0]
ctW = ct.shape[1]
ctr = cv2.resize(ct, dsize=(imgX, imgY), interpolation=cv2.INTER_CUBIC)
ctX, ctY = img_to_lab(ctr)

output = model.predict(ctX)
output *= 128
# #Output colorizations
cur = np.zeros((1,imgX, imgY, 3))
cur[0,:,:,0] = ctX[0,:,:,0]
cur[0,:,:,1:] = output[0]
cur=cur[0]

cur = cv2.resize(cur,dsize=(ctW, ctH), interpolation=cv2.INTER_CUBIC)

plt.figure(figsize=[6,13])
plt.subplot(1,2,1)
plt.imshow(ct)
plt.subplot(1,2,2)
plt.imshow(lab2rgb(cur))
plt.show()
# %%
