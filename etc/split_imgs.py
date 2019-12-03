#%%
import cv2
import glob
from tqdm.auto import tqdm

def split_img(path,x,y,out_folder):
    img = cv2.imread(path)
    cx = path.split("\\")[-1].split('.')[0].split('_')[0]
    core = path.split("\\")[-2]
    for r in tqdm(range(0,img.shape[0],x),position=0):
        for c in range(0,img.shape[1],y):
            cv2.imwrite(out_folder+"\\"+core+"_"+cx+"_"+str(r)+"_"+str(c)+".png",img[r:r+x, c:c+y,:])

foto_folder = 'ct_color\\DADOS_Joao\\ARQUIVOS_HDI'
out_folder = 'ct_color\\split'

n=1
images = glob.glob(foto_folder + "/**/*.png",recursive=True)
for file in images:
    print('Image '+str(n)+' of '+str(len(images)))
    split_img(file,512,512,out_folder)
    n+=1

 # %%


# %%
