# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 09:43:04 2022

@author: 44788
"""
import numpy as np
import pandas as pd
import pickle 
import gzip
import pandas as pd
import numpy as np
import os
import random
import sys
import math
from tensorflow import keras
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#from tensorflow.keras.layers.convolutional import Conv2D
#from tensorflow.keras.layers.convolutional import MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Input, Activation,Flatten
from sklearn.metrics import mean_squared_error
#from keras import backend as K
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import models, layers
from sklearn.metrics import r2_score,mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

file = gzip.open('X.p.gz', 'rb')
X = pickle.load(file)
file.close()
labels=pd.read_csv('DeepSynergy.csv',index_col=0)
#labels = pd.concat([labels, labels]) 
labels=labels[labels.label==1]
synergy=labels.synergy
synergy2=labels.synergy2
idx = np.where(labels['label']!=0)
X=X[idx]

np.save('sampleDS.npy', X[0:50,:])
np.save('sampleDS2.npy', X[len(X):len(X)+50,:])

np.save('sampleDSLabel.npy', synergy[0:50,:])
#---------------------------------------------------------------------


# def normalize(X, means1=None, std1=None, means2=None, std2=None, feat_filt=None, norm='tanh_norm'):
#     if std1 is None:
#         std1 = np.nanstd(X, axis=0)
#     if feat_filt is None:
#         feat_filt = std1!=0
#     X = X[:,feat_filt]
#     X = np.ascontiguousarray(X)
#     if means1 is None:
#         means1 = np.mean(X, axis=0)
#     X = (X-means1)/std1[feat_filt]
#     if norm == 'norm':
#         return(X, means1, std1, feat_filt)
#     elif norm == 'tanh':
#         return(np.tanh(X), means1, std1, feat_filt)
#     elif norm == 'tanh_norm':
#         X = np.tanh(X)
#         if means2 is None:
#             means2 = np.mean(X, axis=0)
#         if std2 is None:
#             std2 = np.std(X, axis=0)
#         X = (X-means2)/std2
#         X[:,std2==0]=0
#         return(X, means1, std1, means2, std2, feat_filt)    
# def concatRefine(x_train,y_train,num):

#     x_new=np.zeros((x_train.shape[0]*2,x_train.shape[1]))
#     y_new=[]
#     i=-1
#     for x,y in zip(x_train,y_train):
#         i=i+1

#         x_new[i,:]=np.concatenate((x[128:256],x[0:128],x[256:num]),axis=0)
#         y_new.append(y)
#         i=i+1
#         x_new[i,:]=x
#         y_new.append(y)
#     return x_new,y_new
# def creatCombUNiq(data):
#     UniqList=[]
#     for d1, d2 in zip(data.drug_a_name,data.drug_b_name):
#         if((d2+'//'+d1) not in UniqList):
#             UniqList.append(d1+'//'+d2)
        
#     UniqList=np.unique(np.array(UniqList))
#     return UniqList
# def toNUMPY(d):
#     data=[]
#     for i in d:
#         data.append(np.array(i))
#     return np.array(data)
# def concatPandas(data,out):
    
#     NewTrain=pd.DataFrame()
#     for d1, d2 in zip(out[:,0],out[:,1]):
#             TrainName=data[(((data.drug_name_a==d1)  &  (data.drug_name_b==d2 ))|((data.drug_name_b==d2)  & (data.drug_name_a==d1 )))]
            

#             NewTrain = pd.concat([NewTrain,TrainName])
#     return NewTrain
# UniqList=creatCombUNiq(labels)
# seedd=94
# data=labels
# y=synergy.values
# kf = KFold(n_splits=5, random_state=seedd, shuffle=True)
# #kf = KFold(n_splits=5)
# kf.get_n_splits(UniqList)
# prenew=np.array([])
# realnew=np.array([])
# ii=0
# num=356
# layers = [8182,4096,1] 
# epochs = 1000 
# act_func = tf.nn.relu 
# dropout = 0.5 
# input_dropout = 0.2
# eta = 0.00001 
# norm = 'tanh'
# outFinal=pd.DataFrame()
# for train_index, test_index in kf.split(UniqList):
#     outPut=pd.DataFrame()

#     ii=ii+1
#     Comb_train, Comb_test = UniqList[train_index], UniqList[test_index]
#     out_train= toNUMPY(np.char.split(Comb_train, sep ='//'))
#     out_test =toNUMPY( np.char.split(Comb_test, sep ='//'))
#     TrainName=concatPandas(data,out_train)
#     TestName=concatPandas(data,out_test)
#     X_test=X[TestName.index.astype(int)]
#     y_test=y[TestName.index.astype(int)]
#     X_train=X[TrainName.index.astype(int)]
#     y_train=y[TrainName.index.astype(int)]
#     X_train, X_val, y_train, y_val = train_test_split( X_train, y_train, test_size=0.20,random_state=seedd)
#     X_train,y_train=concatRefine(X_train,y_train,num)
#     outPut['fold']=np.zeros([len(TestName),])+ii
#     outPut['Index']=TestName.index.astype(int)
#     X_train, mean, std, mean2, std2, feat_filt = normalize(X_train, norm=norm)
#     X_test, mean, std, mean2, std2, feat_filt = normalize(X_test, mean, std, mean2, std2, 
#                                                       feat_filt=feat_filt, norm=norm)
#     model = Sequential()
#     for i in range(len(layers)):
#         if i==0:
#             model.add(Dense(layers[i], input_shape=(X_train.shape[1],), activation=act_func, 
#                             kernel_initializer='he_normal'))
#             model.add(Dropout(float(input_dropout)))
#         elif i==len(layers)-1:
#             model.add(Dense(layers[i], activation='linear', kernel_initializer="he_normal"))
#         else:
#             model.add(Dense(layers[i], activation=act_func, kernel_initializer="he_normal"))
#             model.add(Dropout(float(dropout)))
#         model.compile(loss='mean_squared_error', optimizer=K.optimizers.SGD(lr=float(eta), momentum=0.5))
        
#     hist = model.fit(X_train, y_train, epochs=epochs, shuffle=True, batch_size=64, validation_data=(X_val, y_val))

#     pre_test=model.predict(X_test)
#     prenew=np.concatenate((pre_test,prenew), axis=None)
#     realnew=np.concatenate((y_test,realnew), axis=None)
#     outPut['Real']=y_test
#     outPut['Pre']=pre_test
#     outFinal = pd.concat([outFinal,outPut])

# outFinal.to_csv('Prediction.csv', outFinal)



































#deep synergy
# layers = [8182,4096,1] 
# epochs = 1000 
# act_func = tf.nn.relu 
# dropout = 0.5 
# input_dropout = 0.2
# eta = 0.00001 
# norm = 'tanh'
model = Sequential()
for i in range(len(layers)):
    if i==0:
        model.add(Dense(layers[i], input_shape=(X_tr.shape[1],), activation=act_func, 
                        kernel_initializer='he_normal'))
        model.add(Dropout(float(input_dropout)))
    elif i==len(layers)-1:
        model.add(Dense(layers[i], activation='linear', kernel_initializer="he_normal"))
    else:
        model.add(Dense(layers[i], activation=act_func, kernel_initializer="he_normal"))
        model.add(Dropout(float(dropout)))
    model.compile(loss='mean_squared_error', optimizer=K.optimizers.SGD(lr=float(eta), momentum=0.5))
    
hist = model.fit(X_tr, y_tr, epochs=epochs, shuffle=True, batch_size=64, validation_data=(X_val, y_val))
val_loss = hist.history['val_loss']
model.reset_states()

