import nibabel as nib
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
import datetime
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import UpSampling3D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPool3D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Lambda


from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import AveragePooling3D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

main_path = "~/data2/"

def get_saliency_map(model, image, class_idx):
    with tf.GradientTape() as tape:
        tape.watch(image)
        predictions = model(image)
        
        loss = predictions[:, class_idx]
        
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, image)
    print(gradient.shape)
    # take maximum across channels
    #gradient = tf.reduce_max(gradient, axis=-1)
    
    # convert to numpy
    gradient = gradient.numpy()
    
    # normaliz between 0 and 1
    min_val, max_val = np.min(gradient), np.max(gradient)
    smap = (gradient - min_val) / (max_val - min_val + keras.backend.epsilon())
    
    return smap


def get_img_filenames(main_path):
  patients_list = []
  for root, dirs, files in os.walk(main_path):
    id = len(files)
    for k in range(id):
      if(k%10==0):
        print(k)
      img_name = files[k]
      patients_list.append(img_name)
  return patients_list

def myfun(main_path,ids):
  brains_comb = np.empty(shape=(len(ids),113,137,113,1))
  brains_loc = np.empty(shape=(len(ids),3))
  brains_loc1 = np.empty(shape=(0,3))
  for ind,i in enumerate(ids):   
    if(i.split("--")[0] == "control"):
      brains_loc1 = [0,1,0]
    elif(i.split("--")[0] == "left"):
      brains_loc1 = [1,0,0]
    elif(i.split("--")[0] == "right"):
      brains_loc1 = [0,0,1]
    img = nib.load(main_path+i).get_fdata() # load the subject brain
    img = (img - np.min(img))/(np.max(img)-np.min(img))
    img = np.expand_dims(img, -1)
    brains_comb[ind,:,:,:,:] = img
    brains_loc[ind,:] = brains_loc1
   # print(brains_comb.shape)
   # print(brains_loc.shape)

  return brains_comb, pd.DataFrame(brains_loc,columns = ["Left","Control","Right"])

patients_list = get_img_filenames(main_path)

np.random.seed(1)
np.random.shuffle(patients_list)

labeled_patients = np.append(np.repeat(range(5),len(patients_list)/5),np.repeat(4,2))

def DataGenerator(main_path,patients_list,batch_size):
  while True:
    for batch_index in range(int(len(patients_list)/batch_size)):
      ids = patients_list[batch_index*batch_size:(batch_index+1)*batch_size]
      images, locations = myfun(main_path, ids)
      yield np.array(images), np.array(locations)

batch_size = 16
num_layers = 6
adam = 0.001

inp = tensorflow.keras.Input(shape=(113,137,113, 1), name="input_image")
x = layers.Conv3D(8,(3,3,3), padding = "same", activation="relu", kernel_regularizer = tf.keras.regularizers.l2(l2=0.01))(inp)
x = layers.BatchNormalization()(x)
x = layers.MaxPool3D()(x)
x = layers.Conv3D(16,(3,3,3), padding = "same", activation="relu", kernel_regularizer = tf.keras.regularizers.l2(l2=0.01))(inp)
x = layers.BatchNormalization()(x)
x = layers.MaxPool3D()(x)
x = layers.Conv3D(32,(3,3,3), padding = "same", activation="relu", kernel_regularizer = tf.keras.regularizers.l2(l2=0.01))(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPool3D()(x)
x = layers.Conv3D(64,(3,3,3), padding = "same", activation="relu", kernel_regularizer = tf.keras.regularizers.l2(l2=0.01))(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPool3D()(x)
x = layers.Conv3D(128,(3,3,3), padding = "same", activation="relu", kernel_regularizer = tf.keras.regularizers.l2(l2=0.01))(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPool3D()(x)
x = layers.Conv3D(256,(3,3,3), padding = "same", activation="relu", kernel_regularizer = tf.keras.regularizers.l2(l2=0.01))(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPool3D()(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(3, activation = "softmax", kernel_regularizer = tf.keras.regularizers.l2(l2=0.01))(x)
model = tf.keras.Model(inp, x, name="model")

model.summary()

model_names = ["list with different saved model names"]
for i in model_names:
    if not os.path.exists('~/saliency_maps/'+i+'/'):
        os.makedirs('~/saliency_maps/'+i+'/')
    if not os.path.exists('~/saliency_maps/'+i+'/images/'):
        os.makedirs('~/saliency_maps/'+i+'/images/')    

for model_name in model_names:
    print(model_name)
    for i in range(5):
        print(i)
        test_patients_list = np.array(patients_list)[labeled_patients==i]
        patients_DataGenerator = DataGenerator(main_path,test_patients_list,batch_size=1)
        path = '~/models/'+model_name+'_'+str(batch_size)+'_'+str(num_layers)+'_'+str(adam)+'/Train'+str(i+1)
        checkpoint_filepath = path+'/cp.cpkt'
        model.load_weights(checkpoint_filepath)
        model.layers[-1].activation = tf.keras.activations.linear
        for j in test_patients_list:
            img = next(patients_DataGenerator)[0]
            if(j.split("--")[0] == "control"):
              idx = 1
              saliency_map = get_saliency_map(model, image = tf.constant(img[0:1]), class_idx = idx)
              np.save('~/saliency_maps/'+model_name+'/'+j+'_control.npy',saliency_map)
            elif(j.split("--")[0] == "contol"):
              idx = 1
              saliency_map = get_saliency_map(model, image = tf.constant(img[0:1]), class_idx = idx)
              np.save('~/saliency_maps/'+model_name+'/'+j+'_control.npy',saliency_map)
            elif(j.split("--")[0] == "left"):
              idx = 0
              saliency_map = get_saliency_map(model, image = tf.constant(img[0:1]), class_idx = idx)
              np.save('~/saliency_maps/'+model_name+'/'+j+'_left.npy',saliency_map)
            elif(j.split("--")[0] == "right"):
              idx = 2
              saliency_map = get_saliency_map(model, image = tf.constant(img[0:1]), class_idx = idx)
              np.save('~/saliency_maps/'+model_name+'/'+j+'_right.npy',saliency_map)


for model_name in model_names:
    saliency_map_controls = []
    saliency_map_left = []
    saliency_map_right = []
    for root, dirs, files in os.walk('~/saliency_maps/'+model_name+'/'):
        x = len(files)
        for i in files:
            print(i.split("--")[0])
            if(i.split("--")[0] == "control"):
                saliency_map_controls.append(np.load('~/saliency_maps/'+model_name+'/'+i))
            elif(i.split("--")[0] == "left"):
                saliency_map_left.append(np.load('~/saliency_maps/'+model_name+'/'+i))
            elif(i.split("--")[0] == "right"):
                saliency_map_right.append(np.load('~/saliency_maps/'+model_name+'/'+i))
    control_mean = np.mean(saliency_map_controls,axis=(0,1,5))
    left_mean = np.mean(saliency_map_left,axis=(0,1,5))
    right_mean = np.mean(saliency_map_right,axis=(0,1,5))
    channel = 0
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(np.rot90(img[0,:,60,:,channel]), cmap = 'gray')
    ax.imshow(np.rot90(left_mean[:,60,:]),alpha = 0.4,cmap='jet',vmin=0.2,vmax=0.7)
    plt.savefig('~/saliency_maps/'+model_name+'/images/left.png',bbox_inches = 'tight')
    plt.close(fig)
    channel = 0
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(np.rot90(img[0,:,60,:,channel]), cmap = 'gray')
    ax.imshow(np.rot90(control_mean[:,60,:]), alpha = 0.4,cmap='jet',vmin=0.2,vmax=0.7)
    plt.savefig('~/saliency_maps/'+model_name+'/images/control.png',bbox_inches = 'tight')
    plt.close(fig)
    channel = 0
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(np.rot90(img[0,:,60,:,channel]), cmap = 'gray')
    ax.imshow(np.rot90(right_mean[:,60,:]), alpha = 0.4,cmap='jet',vmin=0.2,vmax=0.7)
    plt.savefig('~/saliency_maps/'+model_name+'/images/right.png',bbox_inches = 'tight')
    plt.close(fig)
