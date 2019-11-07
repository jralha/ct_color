#%%Info for loading
core_folder = 'ct_color\\DADOS_Joao\\_ARQUIVOS_NII'
caixas = ['T1_CX1','T1_CX2','T1_CX3','T1_CX4','T1_CX5','T1_CX6',
          'T2_CX1','T2_CX2','T2_CX3','T2_CX4','T2_CX5','T2_CX6']
slices = [      92,      77,     105,      83,      78,     107,
                78,      77,      75,     106,      76,     108]
data_dict = dict(zip(caixas,slices))

#%%Load CT data
e1=[]
e2=[]
for file in (glob.glob(core_folder + "/**/*.nii",recursive=True)):
    
    fname = file.split('\\')[-1].split('.')[0]
    caixa = fname.split('_')[1]
    core = fname.split('_')[2]
    slc = data_dict[core+'_'+caixa]
    data = nib.load(file).get_data()
    slc0 = np.array(data[slc-1,:,:])
    slc1 = np.array(data[slc-6,:,:])
    slc2 = np.array(data[slc+4,:,:])
    if 'Energy1' in file:
        e1.append(slc0)
        e1.append(slc1)
        e1.append(slc2)
    elif 'Energy2' in file:
        e2.append(slc0)
        e2.append(slc1)
        e2.append(slc2)

e1 = np.array(e1)
e2 = np.array(e2)

Ys = np.vstack((e1,e2))