

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

#Please customize this EXP

def pearson(pred,y):
    pear = stats.pearsonr(y, pred)
    pear_value = pear[0]
    pear_p_val = pear[1]
    print("Pearson correlation is {} and related p_value is {}".format(pear_value, pear_p_val))
    return pear_value

def spearman(pred,y):
    spear = stats.spearmanr(y, pred)
    spear_value = spear[0]
    spear_p_val = spear[1]
    print("Spearman correlation is {} and related p_value is {}".format(spear_value, spear_p_val))
    return spear_value

def DNN(inputShape,n1=1024,n2=1024,n3=500,lr=0.0001):
	model = models.Sequential()
	model.add(layers.Dense(n1,kernel_initializer="he_normal", input_shape=[inputShape]))
	model.add(layers.Dropout(0.5))# % of features dropped)
	#model.add(layers.Dense(1024, activation='relu',kernel_initializer="he_normal"))
	#model.add(layers.Dropout(0.3))# % of features dropped)
	model.add(layers.Dense(n2, activation='relu',kernel_initializer="he_normal"))
	model.add(layers.Dropout(0.3))# % of features dropped)
	model.add(layers.Dense(n3, activation='tanh',kernel_initializer="he_normal"))
	# output layer
	model.add(layers.Dense(1))
	model.compile( optimizer=keras.optimizers.Adam(learning_rate=float(lr),beta_1=0.9, beta_2=0.999, amsgrad=False), loss='mean_squared_error',metrics=['mse', 'mae'])
	return model
#Please customize this EXP
data=pd.read_csv('/homes/rzgar/Narjes/Inputs/DrugComb.csv')
FeatureName=['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5','C1','C2','C3','C4','C5','D1','D2','D3','D4','D5','E1','E2','E3','E4','E5']

def concatRefine(x_train,y_train):

    x_new=np.zeros((x_train.shape[0]*2,x_train.shape[1]))
    y_new=[]
    i=-1
    for x,y in zip(x_train,y_train):
        i=i+1

        x_new[i,:]=np.concatenate((x[128:256],x[0:128],x[256:356]),axis=0)
        y_new.append(y)
        i=i+1
        x_new[i,:]=x
        y_new.append(y)
    return x_new,y_new
def creatCombUNiq(data):
    UniqList=[]
    for d1, d2 in zip(data.Drug1,data.Drug2):
        if((d2+'//'+d1) not in UniqList):
            UniqList.append(d1+'//'+d2)
        
    UniqList=np.unique(np.array(UniqList))
    return UniqList
def concatPandas(data,out):
    
    NewTrain=pd.DataFrame()
    for d1, d2 in zip(out[:,0],out[:,1]):
            TrainName=data[(((data.Drug1==d1)  &  (data.Drug2==d2 ))|((data.Drug1==d2)  & (data.Drug2==d1 )))]
            

            NewTrain = pd.concat([NewTrain,TrainName])
    return NewTrain
def toNUMPY(d):
    data=[]
    for i in d:
        data.append(np.array(i))
    return np.array(data)
def runapp(i,n1,n2,n3,batch,lr,part,tissue=False,cellline=False,fold=1):
    UniqList=np.unique(np.concatenate((np.array(data.Drug1), np.array(data.Drug2)), axis=None))

    #Please customize this EXPs
    X=np.load('/nfs/research/petsalaki/users/rzgar/Inputs/FeaturesR1.npy', mmap_mode='r')[:,:,i]
    y=np.load('/nfs/research/petsalaki/users/rzgar/Inputs/ScoresR1.npy', mmap_mode='r')[:,2]
    kf = KFold(n_splits=5, random_state=94, shuffle=True)
    kf.get_n_splits(UniqList)
    prenew=np.array([])
    realnew=np.array([])
    ii=0
    outFinal=pd.DataFrame()
    for train_index, test_index in kf.split(UniqList):
        outPut=pd.DataFrame()

        ii=ii+1
        if ii==fold:
            if tissue!=False:
                Comb_train, Comb_test = UniqList[train_index], UniqList[test_index]
            
                TrainName=data[(np.logical_not(data.Drug1.isin(Comb_test))  &  np.logical_not(data.Drug2.isin(Comb_test )))]
                TestName=data[((data.Drug1.isin(Comb_test))  |  (data.Drug2.isin(Comb_test )))]

                TrainName=TrainName[TrainName.Tissue!=tissue]
                TestName=TestName[TestName.Tissue==tissue]
                X_test=X[TestName.index.astype(int)]
                y_test=y[TestName.index.astype(int)]
                X_train=X[TrainName.index.astype(int)]
                y_train=y[TrainName.index.astype(int)]
                X_train, X_val, y_train, y_val = train_test_split( X_train, y_train, test_size=0.20,random_state=42)
                X_train,y_train=concatRefine(X_train,y_train)
                outPut['fold']=np.zeros([len(TestName),])+ii
                outPut['Index']=TestName.index.astype(int)
                
                CNN_model=DNN(356,n1,n2,n3,lr)
        
                #Please customize these EXPs
                cb_check = ModelCheckpoint(('/nfs/research/petsalaki/users/rzgar/Outputs/DNN_EXP_Tissue_CV2_'+str(part)+'_'+str(FeatureName[i])+'_Loewe'), verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
        
                y_train=np.array(y_train)
                
                CNN_model.fit(x=X_train,y=y_train,batch_size=batch,epochs = 100,shuffle=True,validation_data = (X_val,y_val),callbacks=[EarlyStopping(monitor='val_loss', mode='auto', patience = 10),cb_check] )
        
                CNN_model = tf.keras.models.load_model('/nfs/research/petsalaki/users/rzgar/Outputs/DNN_EXP_Tissue_CV2_'+str(part)+'_'+str(FeatureName[i])+'_Loewe')
                pre_test=CNN_model.predict(X_test)

                prenew=np.concatenate((pre_test,prenew), axis=None)
                realnew=np.concatenate((y_test,realnew), axis=None)
                outPut['Real']=y_test
                outPut['Pre']=pre_test
                outFinal.to_csv('/nfs/research/petsalaki/users/rzgar/ResultEXP_TissueCV2_'+str(FeatureName[i])+'_'+str(ii)+'_'+str(tissue)+'.csv')

            if cellline!=False:
                Comb_train, Comb_test = UniqList[train_index], UniqList[test_index]
            
                TrainName=data[(np.logical_not(data.Drug1.isin(Comb_test))  &  np.logical_not(data.Drug2.isin(Comb_test )))]
                TestName=data[((data.Drug1.isin(Comb_test))  |  (data.Drug2.isin(Comb_test )))]
                TrainName=TrainName[TrainName.Cell_Line!=cellline]
                TestName=TestName[TestName.Cell_Line==cellline]
                X_test=X[TestName.index.astype(int)]
                y_test=y[TestName.index.astype(int)]
                X_train=X[TrainName.index.astype(int)]
                y_train=y[TrainName.index.astype(int)]
                X_train, X_val, y_train, y_val = train_test_split( X_train, y_train, test_size=0.20,random_state=42)
                X_train,y_train=concatRefine(X_train,y_train)
                outPut['fold']=np.zeros([len(TestName),])+ii
                outPut['Index']=TestName.index.astype(int)
                
                CNN_model=DNN(356,n1,n2,n3,lr)
        
                #Please customize these EXPs
                cb_check = ModelCheckpoint(('/nfs/research/petsalaki/users/rzgar/Outputs/DNN_EXPCV2_Cell_'+str(part)+'_'+str(FeatureName[i])+'_Loewe'), verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
        
                y_train=np.array(y_train)
                
                CNN_model.fit(x=X_train,y=y_train,batch_size=batch,epochs = 100,shuffle=True,validation_data = (X_val,y_val),callbacks=[EarlyStopping(monitor='val_loss', mode='auto', patience = 10),cb_check] )
        
                CNN_model = tf.keras.models.load_model('/nfs/research/petsalaki/users/rzgar/Outputs/DNN_EXPCV2_Cell_'+str(part)+'_'+str(FeatureName[i])+'_Loewe')
                pre_test=CNN_model.predict(X_test)
                prenew=np.concatenate((pre_test,prenew), axis=None)
                realnew=np.concatenate((y_test,realnew), axis=None)
                outPut['Real']=y_test
                outPut['Pre']=pre_test
                outFinal.to_csv('/nfs/research/petsalaki/users/rzgar/ResultEXP_CellCV2_'+str(FeatureName[i])+'_'+str(ii)+'_'+str(cellline)+'.csv')



 
def main():
    i=0
    n1=1024
    n2=1024
    n3=500
    lr=0.0001
    batch=128
    part=1
    tissue=False
    cell=False
    fold=1
    # get the options from user
    for arg in sys.argv[1:]:
      (key,val) = arg.rstrip().split('=')
      if key == 'DrugF':
          i=val
      if key == 'N1':
          n1=val    
      if key == 'N2':
          n2=val   
      if key == 'N3':
          n3=val   
      if key == 'Batch':
          batch=val 
      if key == 'LR':
          lr=val   
      if key == 'Part':
          part=val  
      if key == 'fold':
          fold=val  
      if key == 'cell':
          cell=val  
      if key == 'tissue':
          tissue=val  
    runapp(int(i),int(n1),int(n2),int(n3),int(batch),float(lr),part,tissue,cell,fold)
main()