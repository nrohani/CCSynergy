

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

#Please customize this path

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
#Please customize this path
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
def runapp(i,n1,n2,n3,batch,lr,part):
    f = open("/nfs/research/petsalaki/users/rzgar/Outputs/PATH_Normal_"+str(FeatureName[i])+".txt", "a")
    rmse_mean=0
    sp_mean=0
    pcc_mean=0
    #Please customize this paths
    X=np.load('/nfs/research/petsalaki/users/rzgar/Inputs/FeaturesR2.npy', mmap_mode='r')[:,:,i]
    y=np.load('/nfs/research/petsalaki/users/rzgar/Inputs/ScoresR2.npy', mmap_mode='r')[:,2]
    kf = KFold(n_splits=5, random_state=94, shuffle=True)
    kf.get_n_splits(X)
    prenew=np.array([])
    realnew=np.array([])
    i=0
    outFinal=pd.DataFrame()
    for train_index, test_index in kf.split(X):
        outPut=pd.DataFrame()
        i=i+1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_val, y_train, y_val = train_test_split( X_train, y_train, test_size=0.20,random_state=42)
        X_train,y_train=concatRefine(X_train,y_train)
        outPut['fold']=np.zeros([len(test_index),])+i
        outPut['Index']=test_index
        
        CNN_model=DNN(356,n1,n2,n3,lr)

        #Please customize these paths
        cb_check = ModelCheckpoint(('/nfs/research/petsalaki/users/rzgar/Outputs/DNN_PATHNormal_'+str(part)+'_'+str(FeatureName[i])+'_Loewe'), verbose=1, monitor='val_loss',save_best_only=True, mode='auto')

        y_train=np.array(y_train)
        
        CNN_model.fit(x=X_train,y=y_train,batch_size=batch,epochs = 100,shuffle=True,validation_data = (X_val,y_val),callbacks=[EarlyStopping(monitor='val_loss', mode='auto', patience = 10),cb_check] )

        CNN_model = tf.keras.models.load_model('/nfs/research/petsalaki/users/rzgar/Outputs/DNN_PATHNormal_'+str(part)+'_'+str(FeatureName[i])+'_Loewe')
        pre_test=CNN_model.predict(X_test)
        pcc=pearson(pre_test[:,0],y_test)
        sp=spearman(pre_test,y_test)
        rmse=math.sqrt( mean_squared_error(y_test ,pre_test))
        rmse_mean=rmse+rmse_mean
        sp_mean=sp+sp_mean
        pcc_mean=pcc_mean+pcc
        prenew=np.concatenate((pre_test,prenew), axis=None)
        realnew=np.concatenate((y_test,realnew), axis=None)
        outPut['Real']=y_test
        outPut['Pre']=pre_test
        outFinal = pd.concat([outFinal,outPut])



#    np.save('/nfs/research/petsalaki/users/rzgar/Normal_PATH_Pre_'+str(FeatureName[i])+'.npy', prenew)	
#
#    np.save('/nfs/research/petsalaki/users/rzgar/Normal_PATH_Real_'+str(FeatureName[i])+'.npy', realnew)	
    outFinal.to_csv('/nfs/research/petsalaki/users/rzgar/Result_NormalCV_PATH_'+str(FeatureName[i])+'.csv')
    rmse_mean=rmse_mean/5
    sp_mean=sp_mean/5
    pcc_mean=pcc_mean/5
    f.write(str(FeatureName[i])+ '  sp: '+str(sp)+' rmse:'+str(rmse)+' pc: '+str(pcc))
    f.write('\n')
    f.write('Metrics after concat:')
    f.write('\n')
    pcc=pearson(prenew[:,0],realnew)
    sp=spearman(prenew[:,0],realnew)
    rmse=math.sqrt( mean_squared_error(prenew[:,0],realnew))
    f.write(str(FeatureName[i])+ '  sp: '+str(sp)+' rmse:'+str(rmse)+' pc: '+str(pcc))
    f.write('\n')
    f.flush()
 
    f.close()    
def main():
    i=0
    n1=1024
    n2=1024
    n3=500
    lr=0.0001
    batch=128
    part=1
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
    runapp(int(i),int(n1),int(n2),int(n3),int(batch),float(lr),part)
main()