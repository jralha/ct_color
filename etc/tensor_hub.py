#%%Libraries
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#%% Load data and set layers from VGG net
#######################################################################
def load_input_image(path):
    img = Image.open(path)
    img = np.float32(np.array(img))
    img = img[:,:,:3]
    img = img / 255
    img = img[tf.newaxis, :]
    return img

foto = 'ct_color\\reg_foto.png'
ct = 'ct_color\\dados0\\Tomografia_Cx6_T2.PNG'

content_image = load_input_image(ct)
style_image = load_input_image(foto)

#%%
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

#%%
outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
stylized_image = outputs[0]

# %%
plt.figure(figsize=[10,20])
plt.subplot(1,3,1)
plt.imshow(stylized_image[0])
plt.subplot(1,3,2)
plt.imshow(content_image[0])
plt.subplot(1,3,3)
plt.imshow(style_image[0])

# %%
