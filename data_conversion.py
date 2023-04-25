import scipy
import os
import numpy as np
import nibabel as nib
import pandas as pd

directory = '~/data/'

patients_df = pd.read_excel('~/<excel_file>.xlsx')


my_df = patients_df[(patients_df['Side of Epilepsy']==1)|(patients_df['Side of Epilepsy']==2)][['New ID','Side of Epilepsy','Site']]

my_df.index = range(my_df.shape[0])

###################################
###########PATIENTS################
###################################

y = []
dest_path = "~/data2/"
tmp_path = "~/tmp_data/"

for i in range(my_df.shape[0]):
  if(my_df['Side of Epilepsy'][i]==1):
    side = 'left'
  elif(my_df['Side of Epilepsy'][i]==2):
       side='right'
  img_name = side+'--'+my_df['Site'][i].lower().replace(' ','')+"--"+my_df['New ID'][i].lower().replace(' ','')
  my_path = directory+my_df['New ID'][i].replace('_','')+'.mat'
  print(my_df['New ID'][i])
  if(os.path.exists(my_path)):
    mat = scipy.io.loadmat(my_path)
  else:
    my_path = directory+my_df['New ID'][i].replace('_','')+'MissingLesion.mat'
    mat = scipy.io.loadmat(my_path)
  if(('pre' in mat.keys())==True):
    tst = mat['pre']["vbm_gm"].item()["dat"].item()
  else:
    tst = mat['pos']["vbm_gm"].item()["dat"].item()
  np.save(tmp_path +my_df['New ID'][i], tst)
  d_load = np.load(tmp_path + my_df['New ID'][i]+'.npy',allow_pickle = True)
  data = np.array(d_load, dtype=np.float32)
  affine = np.eye(4)
  nifti_data = nib.Nifti1Image(data, affine)
  nib.save(nifti_data, dest_path + img_name + '.nii')
  if(tst.shape!=(113,137,113)):
    y.append[i]


##########################
##########CONTROL#########
##########################

controls_df = pd.read_excel('~/<excel_file>.xlsx','Controls')


y = []
for i in range(controls_df.shape[0]):
  img_name = "control--"+controls_df['Site'][i].lower().replace(' ','')+"--"+controls_df['New ID'][i].lower().replace(' ','')
  my_path = directory+controls_df['New ID'][i].replace('_','')+'.mat'
  if(os.path.exists(my_path)):
    mat = scipy.io.loadmat(my_path)
    tst = mat['session']["vbm_gm"].item()["dat"].item()
    np.save(tmp_path +controls_df['New ID'][i], tst)
    d_load = np.load(tmp_path + controls_df['New ID'][i]+'.npy',allow_pickle = True)
    data = np.array(d_load, dtype=np.float32)
    affine = np.eye(4)
    nifti_data = nib.Nifti1Image(data, affine)
    nib.save(nifti_data, dest_path + img_name + '.nii') 
  else:
    my_path = directory+patients_df['New ID'][i].replace('_','')+'MissingLesion.mat'
    mat = scipy.io.loadmat(my_path)
    np.save(tmp_path+controls_df[i], mat)
    d_load = np.load(tmp_path+ controls_df[i]+'.npy')
    data = np.array(d_load, dtype=np.float32)
    affine = np.eye(4)
    nifti_data = nib.Nifti1Image(data, affine)
    nib.save(nifti_data, dest_path + img_name + '.nii')
  tst = mat['session']["vbm_gm"].item()["dat"].item()
  if(tst.shape!=(113,137,113)):
    y.append[i]
