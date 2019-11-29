#%%
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#%%
nii_file1 = 'ct_color\\DADOS_Joao\\_ARQUIVOS_NII\\Testemunho 2\\Libra_CX6_T2_231_216_1426_Energy1.nii'
nii_file2 = 'ct_color\\DADOS_Joao\\_ARQUIVOS_NII\\Testemunho 2\\Libra_CX6_T2_231_216_1426_Energy2.nii'

model0 = tf.keras.models.load_model('ct_color\\style_results\\model0.h5')
model1 = tf.keras.models.load_model('ct_color\\style_results\\model1.h5')

#%%
nii1 = np.array(nib.load(nii_file1).get_data())
nii2 = np.array(nib.load(nii_file2).get_data())
# %%
slc = nii1[:,108,:][:,::-1].T
slc = np.expand_dims(slc,axis=2)
slc = slc/slc.max()

ct = np.concatenate((slc,slc,slc),axis=2)
ct = np.expand_dims(ct,axis=0)

#%%
out = model.predict(ct)

#%%
plt.figure(figsize=[10,20])
plt.subplot(1,2,1)
plt.imshow(ct[0])
plt.subplot(1,2,2)
plt.imshow(out[0][0][0])


# %%
