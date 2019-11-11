#%%Libraries
#######################################################
import os
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm.auto import tqdm
import glob
from skimage.color import rgb2lab, lab2rgb

#%%Paths
#######################################################
foto_folder = 'ct_color\\split'
all_photos = glob.glob(foto_folder + "/**/*.png",recursive=True)[0:50]
vgg = tf.keras.applications.vgg19.VGG19(include_top=False)

#%%Functions
#######################################################
def load_img(path):
    imageArray = cv2.imread(path)[:,:,:3]
    return imageArray

def img_to_lab(image):
    ab = rgb2lab(1.0/255*image)[:,:,:3]
    height=image.shape[0]
    width=image.shape[1]
    ab = ab.reshape(1, height, width, 3)

    return ab


#%%Building Input Array in LAB Color Space
#######################################################
test_img = load_img(all_photos[0])
imgX = test_img.shape[0]
imgY = test_img.shape[1]
X = load_img('ct_color\\dados0\\Tomografia_Cx6_T2.PNG')
X = img_to_lab(X)
Y = np.zeros((len(all_photos),imgX,imgY,3))

for pos,foto in tqdm(enumerate(all_photos),position=0):
    img = load_img(foto)
    if img.shape[0] == imgX and img.shape[1] == imgY:
        Y[pos] = img_to_lab(img)
    else:
        img = cv2.resize(img, dsize=(imgX, imgY), interpolation=cv2.INTER_CUBIC)
        Y[pos] = img_to_lab(img)

content_image = X
style_image = Y

#%% Setting content and style layers
########################################################
style_layers=[]
content_layers=[]
for layer in vgg.layers:
    if 'conv1' in layer.name:
        style_layers.append(layer.name)
    elif 'conv2' in layer.name:
        content_layers.append(layer.name)
content_layers = [content_layers[-1]]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

#%% Building the style extraction model
#####################################################################
def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
  
    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float64)
    return result/(num_locations)

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg =  vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                      outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                      for style_output in style_outputs]

        content_dict = {content_name:value 
                      for content_name, value 
                      in zip(self.content_layers, content_outputs)}

        style_dict = {style_name:value
                    for style_name, value
                    in zip(self.style_layers, style_outputs)}
    
        return {'content':content_dict, 'style':style_dict}

extractor = StyleContentModel(style_layers, content_layers)

results = extractor(tf.constant(content_image))

style_results = results['style']

# %% Run Gradient Descent
####################################################################################
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

image = tf.Variable(content_image)

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

style_weight=1e-2
content_weight=1e4

def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

#%% Optimization
total_variation_weight=10

@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += total_variation_weight*tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


image = tf.Variable(content_image)
import time
start = time.time()

epochs = 10
steps_per_epoch = 10

step = 0
for n in range(epochs):
    print("Train epoch: {}".format(n))
    for m in tqdm(range(steps_per_epoch),position=0):
        step += 1
        train_step(image)
        #print(".", end='')
  

end = time.time()
print("Total time: {:.1f}".format(end-start))

# %%Save
original_size = X.shape[1:3]
resize = tf.image.resize(image,original_size)

out = lab2rgb((resize.numpy()[0]*255))
ori = lab2rgb(X[0])


plt.figure(figsize=[10,40])
plt.subplot(1,2,1)
plt.imshow(ori)
plt.subplot(1,2,2)
plt.imshow(out)
plt.show()

# %%
