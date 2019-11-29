#%%
import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import glob

photo_folder = 'ct_color\\DADOS_Joao\\ARQUIVOS_HDI'

foto_folder = 'ct_color\\split'
all_photos = glob.glob(photo_folder + "/**/*.png",recursive=True)[0:50]

 # %%


# %%
