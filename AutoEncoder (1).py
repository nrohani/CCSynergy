from keras.layers import Input, Dense, Layer, InputSpec
from keras import regularizers, activations, initializers, constraints, Sequential
from keras.models import Model, load_model
import numpy as np
import pandas as pd

import sys

def Autoencoder(path,input_dim, encoding_dim,learning_rate,nb_epoch,batch_size,activF,opt='sgd'):
    

    data=pd.read_csv(path)
    encoder = Dense(encoding_dim, activation=activF, input_shape=(input_dim,), use_bias = True) 
    decoder = Dense(input_dim, activation=activF, use_bias = True)
    
    autoencoder = Sequential()
    autoencoder.add(encoder)
    autoencoder.add(decoder)
    
    autoencoder.compile(metrics=['accuracy'],
                    loss='mean_squared_error',
                    optimizer=opt)
    
    autoencoder.fit(data, data,
                epochs=nb_epoch,
                batch_size=batch_size,
                shuffle=True,
                verbose=0)
    encoder_layer = Model(inputs=autoencoder.inputs, outputs=autoencoder.layers[0].output)
    encoded_features = np.array(encoder_layer.predict(data))
    return encoded_features
def main():
    opt='sgd'
    encD=100
    lr=0.00001
    epoch=100
    activF='linear'
    path='data/'
    batch=100
    input_dim=100
    # get the options from user
    for arg in sys.argv[1:]:
      (key,val) = arg.rstrip().split('=')
      if key == 'opt':
          opt=val
      if key == 'encoding_dim':
          encD=val    
      if key == 'lr':
          lr=val   
      if key == 'epoch':
          epoch=val   
      if key == 'batch':
          batch=val 
      if key == 'activF':
          activF=val   
      if key == 'dataPath':
          path=val  
      if key == 'inputDim':
          input_dim=val 
    Ex_features=Autoencoder(path,input_dim, encD,lr,epoch,batch,activF,opt)
    pd.DataFrame(Ex_features).to_csv('Features.csv')
#def main():
#    opts=['sgd','adam']
#    enDims=[10,20,30,40,50,60]
#    lrs=[0.0001,0.00001,0.001]
#    epochs=[10,100,50,200]
#    activs=['linear', 'relu', 'sigmoid']
#    batchs=[100,50,200,300]
#    