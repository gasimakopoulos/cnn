import nibabel as nib
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import tensorflow
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


main_path = "~/data2/"

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

filter_layers = [[8,16,32,64,128,256],[8,16,32,64,128,256,512,512],[8,16,32,64,128,256,512,512,512,512]]
learning_rate = [0.01,0.001,0.0001]
batch_sizes = [8,16,32]

i=0

model_name = 'simple_model'

for batch_size in batch_sizes:
    for adam in learning_rate:
        for filters in filter_layers:
            inp = tensorflow.keras.Input(shape=(113,137,113, 1), name="input_image")
            x=inp
            print('batch_size=',batch_size,'adam = ',adam,'filters = ',filters)
            for num_filter in filters:
                x = layers.Conv3D(num_filter,(3,3,3), padding = "same", activation="relu", kernel_regularizer = tf.keras.regularizers.l2(l2=0.01))(x)
                x = layers.BatchNormalization()(x)
                if(x.shape[1]>1 and x.shape[2]>1 and x.shape[3]>1):
                    x = layers.MaxPool3D()(x)
            x = layers.Flatten()(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(3, activation = "softmax", kernel_regularizer = tf.keras.regularizers.l2(l2=0.01))(x)
            model = tf.keras.Model(inp, x, name="model")
            model.summary()
            
            model.compile(
              optimizer=tf.keras.optimizers.Adam(learning_rate=adam),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy']
            )

            path = '~/models/'+model_name+'_learningrate'+str(adam)+'_batchsize'+str(batch_size)+'filters'+str(len(filters))+'/Train'+str(i+1)
            res_path = '~/models/'+model_name+'_learningrate'+str(adam)+'_batchsize'+str(batch_size)+'filters'+str(len(filters))+'/Train'+str(i+1)+'/results'
            if not os.path.exists(path):
              os.makedirs(path)
            if not os.path.exists(res_path):
              os.makedirs(res_path)
            test_patients_list = np.array(patients_list)[labeled_patients==i]
            train_and_val_list =  np.array(patients_list)[labeled_patients!=i]
            train_patients_list, val_patients_list =  train_and_val_list[ : (int(len(train_and_val_list) * .8)+1)], train_and_val_list[(int(len(train_and_val_list) * .8)+1):]
            training_DataGenerator = DataGenerator(main_path,train_patients_list,batch_size= batch_size)
            val_DataGenerator = DataGenerator(main_path,val_patients_list,batch_size=batch_size)
            checkpoint_filepath = path+'/cp.cpkt'
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=True,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,verbose=1)

            es = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss',
                                          min_delta=1e-4,
                                          patience=20,
                                          verbose=0, mode='auto')


            logdir = path+os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)


            model.fit(training_DataGenerator,validation_data = val_DataGenerator,
                      steps_per_epoch = int(len(train_patients_list)/batch_size),
                      validation_steps=int(len(val_patients_list)/batch_size),epochs = 1000,
                      verbose=1,callbacks=[model_checkpoint_callback,tensorboard_callback,es])

            model.load_weights(checkpoint_filepath)

            all_pred, all_y = [],[]
            for j in test_patients_list:
              datagen_val = DataGenerator(main_path,[j],batch_size=1)
              x,y = next(datagen_val)
              all_pred.append(model.predict(x)[0])
              all_y.append(np.array(y)[0])
            all_pred_classes = np.argmax(all_pred,axis = 1)
            all_y_classes = np.argmax(all_y,axis = 1)
            pd.crosstab(all_y_classes,all_pred_classes)
            matrix = confusion_matrix(all_pred_classes,all_y_classes)
            CM = pd.DataFrame(matrix).transpose()
            CM.to_csv(res_path+'/CM.csv')

            matrix = classification_report(all_pred_classes,all_y_classes,output_dict=True)
            CR = pd.DataFrame(matrix).transpose()
            CR.to_csv(res_path+'/CR.csv')
