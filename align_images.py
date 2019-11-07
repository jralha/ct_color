#%% Libraries
#####################################
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#%% Loading Data
#####################################
foto = cv2.imread('ct_color\\caixa6_T2_invertido.png')
ct = cv2.imread('ct_color\\Tomografia_Cx6_T2.PNG')


#%%Plot both for picking
def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata

    global coords
    coords.append((ix, iy))

    if len(coords) == 2:
        fig.canvas.mpl_disconnect(cid)

    return coords

coords=[]
fig,ax = plt.subplots(figsize=[10,30])
ax.imshow(ct)
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
pts_dst=np.array(coords)
print(pts_dst)

coords=[]
fig,ax = plt.subplots(figsize=[20,30])
ax.imshow(foto)
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
pts_src=np.array(coords)
print(pts_src)

print(len(pts_src)==len(pts_src))

#%%Transform Photo to CT scale and lines up images
# Find homography
h, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC)
 
# Use homography
height, width, channels = ct.shape
fotoReg = cv2.warpPerspective(foto, h, (width, height))

plt.figure(figsize=[10,30])
plt.subplot(1,2,1)
plt.imshow(fotoReg)
plt.subplot(1,2,2)
plt.imshow(ct)
plt.show()

cv2.imwrite('ct_color\\reg_foto.png',fotoReg)



# %%
