# health_ai
# keep this code

import numpy as np
from random import randint
from sklearn.utils import shuffle 
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
#from tensorflow import keras
import keras as kr
from keras.models import Sequential 
from keras.optimizers import Adam 
from keras.layers import Activation, Dense
from keras.metrics import categorical_crossentropy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import os.path
import pandas as pd
from keras.models import load_model
import tensorflow as tf
from sklearn import *



def main():
   dataset_train = pd.read_excel('hotspot-NN.xls')
   training_set = dataset_train.iloc[:, 1:].values
  # dataset_train.head()
   #print (dataset_train)
   #print (training_set)
   training_set = np.transpose(training_set)
   print (training_set)
   print (tf.__version__)
   

   sc = MinMaxScaler(feature_range=(0,1))
   
   training_set_scaled = sc.fit_transform(training_set)
   
   x_train = []
   y_train = []
   
   for i in range(3, 12):
       x_train.append(training_set_scaled[i-3:i, 0]) 
       y_train.append(training_set_scaled[i, 0])

   x_train, y_train = np.array(x_train), np.array(y_train) 
   x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
   
   from keras.models import Sequential 
   from keras.layers import LSTM 
   from keras.layers import Dropout 
   from keras.layers import Dense 

   model = Sequential() 

   model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1))) 
   model.add(Dropout(0.2)) 

   model.add(LSTM(units = 50, return_sequences = True))
   model.add(Dropout(0.2)) 

   model.add(LSTM(units = 50, return_sequences = True))
   model.add(Dropout(0.2)) 

   model.add(LSTM(units = 50, return_sequences = False)) 
   model.add(Dropout(0.2))

   model.add(Dense(units = 5))

   model.compile(optimizer = 'adam', loss = 'mean_squared_error')
   model.fit(x_train, y_train, epochs = 100, batch_size = 4) 


   dataset_test = pd.read_excel('hotspot-NN.xls')
   real_data = dataset_test.iloc[:, 1:].values 

   
   dataset_total = pd.concat((dataset_train, dataset_test), axis = 0) #***
   inputs = dataset_total[len(dataset_total) - len(dataset_test) - 3:].values 

   inputs = inputs.reshape(-1,1)
   inputs = sc.fit_transform(inputs) #***

   x_test = []

   for i in range(3, 12):
      x_test.append(inputs[i-3:i, 0]) 

   x_test = np.array(x_test) 
   x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

   predicted_data = model.predict(x_test)
   predicted_data = sc.inverse_transform(predicted_data) 

   #plt.plot(real_data, color = 'black', label = 'crime rate prediction') 
   plt.plot(predicted_data, color = 'green', label ='crime rate', marker = 'o',
          markerfacecolor = 'red', markersize = 6 )
   
   
   plt.title('crime rate prediction') 
   
   
   #plt.ylim(100, 420)
   #plt.xlim(1, 16)

   plt.xlabel('days')
   plt.ylabel('crimes')
   plt.legend()
   plt.show()








if __name__ == '__main__':
    main()
