#%% Libraries
#####################################
from PIL import Image
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
Image.MAX_IMAGE_PIXELS = 1000000000  

#%% Loading Data
#####################################
foto = Image.open('ct_color\\output0\\reg_foto.png')
ct = Image.open('ct_color\\dados0\\Tomografia_Cx6_T2.PNG')

foto_array = np.asarray(foto)
ct_array = np.asarray(ct)

#%%
x = int(ct_array.shape[0]/(foto_array.shape[0]/foto_array.shape[1]))
y = ct_array.shape[0]

crop_ct = ct.crop((0,0,x,y-7))

size = (x,y)

foto_resize = foto.resize(size,Image.ANTIALIAS)
ct_resize = crop_ct.resize(size,Image.ANTIALIAS)

cropx1, cropx2, cropy1, cropy2 = [0, x, 0, y]

recrop_foto = foto_resize.crop((cropx1, cropy1, cropx2, cropy2))
recrop_ct = ct_resize.crop((cropx1, cropy1, cropx2, cropy2))


ct_in = np.asarray(recrop_ct).T[:-1].T
foto_in = np.asarray(recrop_foto).T[:-1].T

ct_in_flat = ct_in.reshape((ct_in.shape[0]*ct_in.shape[1]),ct_in.shape[2])
foto_in_flat = foto_in.reshape((foto_in.shape[0]*foto_in.shape[1]),foto_in.shape[2])

#%%
X_train, X_test, Y_train_rgb, Y_test_rgb = train_test_split(ct_in_flat,foto_in_flat)

Y_train_r = Y_train_rgb.T[0].T
Y_train_g = Y_train_rgb.T[1].T
Y_train_b = Y_train_rgb.T[2].T
Y_test_r = Y_test_rgb.T[0].T
Y_test_g = Y_test_rgb.T[1].T
Y_test_b = Y_test_rgb.T[2].T

est = xgb.XGBRegressor()
est_r = est
est_g = est
est_b = est

est_r.fit(X_train,Y_train_r)
est_g.fit(X_train,Y_train_g)
est_b.fit(X_train,Y_train_b)

pred_r = est_r.predict(X_test)
pred_g = est_g.predict(X_test)
pred_b = est_b.predict(X_test)

mse_r = mean_squared_error(Y_test_r,pred_r)
mse_g = mean_squared_error(Y_test_g,pred_g)
mse_b = mean_squared_error(Y_test_b,pred_b)
msv_m = np.mean([mse_r,mse_g,mse_b])

ct_r = est_r.predict(ct_in_flat)
ct_g = est_g.predict(ct_in_flat)
ct_b = est_b.predict(ct_in_flat)

color_ct = np.zeros(ct_in.T.shape)
color_ct[0] = ct_r.reshape(ct_in.T.shape[1],ct_in.T.shape[2])
color_ct[1] = ct_g.reshape(ct_in.T.shape[1],ct_in.T.shape[2])
color_ct[2] = ct_b.reshape(ct_in.T.shape[1],ct_in.T.shape[2])
color_ct = color_ct.T.astype(int)

plt.figure(figsize=[10,20])
plt.subplot(1,3,1)
plt.imshow(foto_in)
plt.subplot(1,3,2)
plt.imshow(ct_in)
plt.subplot(1,3,3)
plt.imshow(color_ct)


# %%
