#%%
import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

#%% Load data and set layers from VGG net
#######################################################################
def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def load_input_image(path,shape):
  img = Image.open(path)
  img = img.crop(shape)
  img = np.array(img, dtype=float)
  if img.shape[2] == 4:
    img = img[:,:,:-1]
    
 

  vggsize=(224,224)
  img = tf.image.resize(img, vggsize)
  img = img[tf.newaxis, :]
  return img


foto = 'ct_color\\reg_foto.png'
ct = 'ct_color\\Tomografia_Cx6_T2.PNG'
vgg = tf.keras.applications.vgg19.VGG19(include_top=False)

cropping=(0,380,96,630)
content_image = load_input_image(ct,cropping)
style_image = load_input_image(foto,cropping)
original_size = tf.cast(tf.shape(tf.image.convert_image_dtype(tf.image.decode_image(tf.io.read_file(ct), channels=3), tf.float32))[:-1], tf.float32)
original_size = tuple(map(int,original_size.numpy().tolist()))

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

#%% Building the model
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
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
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

epochs = 1000
steps_per_epoch = 100

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
resize = tf.image.resize(image,original_size)

out = tensor_to_image(resize)
# plt.imshow(out)
out.save('ct_color\\ct_color_style_transfer.png')

# %%Plot  
%matplotlib qt
plt.figure(figsize=[20,20])
plt.subplot(1,4,1)
plt.imshow(PIL.Image.open('ct_color\\Tomografia_Cx6_T2.PNG'))
plt.subplot(1,4,2)
plt.imshow(PIL.Image.open('ct_color\\caixa6_T2_invertido.png'))
plt.subplot(1,4,3)
plt.imshow(PIL.Image.open('ct_color\\reg_foto.png'))
plt.subplot(1,4,4)
plt.imshow(PIL.Image.open('ct_color\\ct_color_style_transfer.png'))
plt.show()

# %%