"""CNN based on all images"""
#%%Libraries

#Standard libraries
import glob
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#General
from tqdm.auto import tqdm
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

#Keras
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, InputLayer
from tensorflow.keras.models import Sequential
tf.autograph.set_verbosity(0)


#Handling images
import cv2

#%%Funstions and settings paths
#Fun
def load_img(path,rX=None,rY=None, rotate=False):
    imageArray = cv2.imread(path)[:,:,:3]

    if rotate == True:
        imageArray = cv2.rotate(imageArray,cv2.ROTATE_90_CLOCKWISE)

    if rX != None and rY != None:
        imageArray = cv2.resize(imageArray, dsize=(rX, rY), interpolation=cv2.INTER_CUBIC)
    imageArray = imageArray/255
    imageArray = imageArray[tf.newaxis, :]
    return imageArray

def load_nii(path,slice_window=None,slice_dim=None,rX=None,rY=None,rotate=False,flipX=False,flipY=False):
    dataArray = np.array(nib.load(path).get_data())

    if slice_dim == 0:
        imageArray = dataArray[slice_window,:,:]
    elif slice_dim == 1:
        imageArray = dataArray[:,slice_window,:]
    elif slice_dim == 2:
        imageArray = dataArray[:,:,slice_window]

    imageArray = cv2.cvtColor(imageArray,cv2.COLOR_GRAY2BGR)

    if rotate == True:
        imageArray = cv2.rotate(imageArray,cv2.ROTATE_90_CLOCKWISE)

    if flipY == True:
        imageArray = imageArray[::-1,:,:]

    if flipX == True:
        imageArray = imageArray[:,::-1,:]

    if rX != None and rY != None:
        imageArray = cv2.resize(imageArray, dsize=(rX, rY), interpolation=cv2.INTER_CUBIC)

    imageArray = imageArray/np.max(imageArray)
    imageArray = imageArray[tf.newaxis, :]
    return imageArray




#%% Setting paths
#Folder for photos
foto_folder = 'ct_color\\DADOS_Joao\\ARQUIVOS_HDI'
ct_base_folder = 'ct_color\\DADOS_Joao\\_ARQUIVOS_NII'
all_photos = glob.glob(foto_folder + "\\*\\*.png",recursive=True)
all_cts = []
slices = []

#Get list of matching CTs
for photo in all_photos:
    testo = photo.split('\\')[-2]
    t = "T"+testo[-1]
    e = 'Energy1'
    caixa = photo.split('\\')[-1].split('.')[-2].split('_')[-2]
    caixa = "CX"+caixa[-1]
    slc = int(photo.split('\\')[-1].split('.')[-2].split('_')[-1])

    cts = glob.glob(ct_base_folder + "\\*\\*.nii",recursive=True)

    for ct in cts:
        if t in ct and caixa in ct and e in ct:
            all_cts.append(ct)

    slices.append(slc)


#%% Load data
print('Loading training images...')

test_img = load_nii(all_cts[0],slice_window=slices[0],slice_dim=1,
                    rotate=True,flipX=False,flipY=True)
imgY = test_img.shape[1]-5
imgX = test_img.shape[2]-5

X = np.zeros((len(all_cts),imgY,imgX,3))
Y = np.zeros((len(all_photos),imgY,imgX,3))

nfoto=1
plot_pos = 1
for pos,foto in enumerate(all_photos):
    print(nfoto)

    Y[pos] = load_img(all_photos[pos],rY=imgY,rX=imgX,rotate=True)
    X[pos] = load_nii(all_cts[pos],slice_window=slices[pos],
              slice_dim=1,rX=imgX,rY=imgY,
              rotate=True,flipX=True,flipY=True)
    nfoto=nfoto+1

    # plt.figure(figsize=[20,100])
    # plt.subplot(12,2,plot_pos)
    # plot_pos+=1
    # plt.imshow(Y[pos])
    # plt.subplot(12,2,plot_pos)
    # plot_pos+=1
    # plt.imshow(X[pos])


#%%# Building the neural network
print('Compiling Model...')
model = Sequential()
model.add(InputLayer(input_shape=(imgY, imgX, 3)))
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
model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))

#%%
model.compile(optimizer='adam', loss='mse')
# %%# Finish model
print('Training model...')
model.fit(x=X,y=Y,batch_size=3,epochs=100,verbose=3)
     


# %%
print('Making prediction...')
ct = X[5][tf.newaxis, :]

output = model.predict(ct)

plt.figure(figsize=[6,13])
plt.subplot(1,2,1)
plt.imshow(ct[0])
plt.subplot(1,2,2)
plt.imshow(output[0])
plt.show()
# %%
