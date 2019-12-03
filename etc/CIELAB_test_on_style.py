#%%
import cv2
import matplotlib.pyplot as plt
import numpy as np
#%%
sty = cv2.cvtColor(cv2.imread('ct_color\\style_results_bkp\\epoch_34199.png'),cv2.COLOR_BGR2LAB)
ct = cv2.cvtColor(cv2.imread('ct_color\\ct.PNG'),cv2.COLOR_BGR2LAB)

# %%
sty_l, sty_a, sty_b = cv2.split(sty)
ct_l, ct_a, ct_b = cv2.split(ct)

wct = 0.4
wsty = 0.6
weighted_l = ((wct * ct_l)+(wsty * sty_l)).astype('uint8')

merged = cv2.merge((weighted_l, sty_a, sty_b))



merged_rgb = cv2.cvtColor(cv2.cvtColor(merged,cv2.COLOR_LAB2BGR),cv2.COLOR_BGR2RGB)

plt.figure(figsize=[10,20])
plt.imshow(merged_rgb)

# %%
