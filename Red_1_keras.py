#%%
#Librerias
#importamos librerias y funciones de tensorflow asi como numpy
import numpy as np 
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras import regularizers
dataset=mnist.load_data()
#%%
#definimos los hiperparametros de nuestra red
learning_rate = 0.01
momentum=0.1
epochs = 30
batch_size = 10
num_classes=10
#%%
# hacemos tuplas de nuestros datos de entrenemiento y nuestros datos de prueva 
# a continuacion le damos una nueva estructura a los datos de entrada, para poder usar una red con el numero de entradas
#igual al numero de pixeles de nuestras imagenes, definimoslos datos como tipo float 32
# y los normalizamos dividiendo entre 255.
(x_tr, y_tr), (x_t, y_t)=dataset
x_trv=x_tr.reshape(60000,784)
x_tv=x_t.reshape(10000,784)
x_trv=x_trv.astype('float32')
x_tv=x_tv.astype('float32')
x_tv /= 255
x_trv /= 255
# %%
#le damos formato one hot a los vectores de salida para poder usarlos de manera mas eficiente
y_trc=keras.utils.to_categorical(y_tr, num_classes)
y_tc=keras.utils.to_categorical(y_t, num_classes)
# %%
Capa_salida=Dense(num_classes, activation='sigmoid')
model = Sequential()
model.add(Dense(30, activation='sigmoid', input_shape=(784,)))
model.add(Capa_salida)
#model.add(Dense(num_classes,activation='softmax'))
model.summary()
#%%
model.compile(loss='categorical_crossentropy',optimizer=SGD(learning_rate=learning_rate, momentum=momentum),metrics=['accuracy'])
# %%
history = model.fit(x_trv, y_trc,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_tv, y_tc)
                    )
# %%
