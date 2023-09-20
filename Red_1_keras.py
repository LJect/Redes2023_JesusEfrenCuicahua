
#%%
# Librerias
#importamos librerias y funciones de tensorflow asi como numpy
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras.regularizers import l1, l2
from keras.callbacks import ModelCheckpoint
#%%
dataset=mnist.load_data()
#%%
#Experimento = comet_ml.Experiment(
 #   auto_histogram_weight_logging=True,
  #  auto_histogram_gradient_logging=True,
   # auto_histogram_activation_logging=True,
    #log_code=True,
    # )
#%%
#definimos los hiperparametros de nuestra red
parametros = {
    "batch_size": 20,
    "learning_rate":0.0002,
    "beta_1":0.01,
    "epochs": 60,
    "momentum":0.1,
    "num_classes": 10,
    "loss": "categorical_crossentropy",
}
#Experimento.log_parameters(parametros)
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
y_trc=keras.utils.to_categorical(y_tr, parametros['num_classes'])
y_tc=keras.utils.to_categorical(y_t, parametros['num_classes'])
# %%
Capa_salida=Dense(parametros['num_classes'], activation='sigmoid')
model = Sequential()
model.add(Dense(80, activation='sigmoid', input_shape=(784,),kernel_regularizer=l1(0.0001)))
model.add(Dropout(0.2))
model.add(Dense(40, activation='sigmoid', kernel_regularizer=l2(0.0001)))
model.add(Capa_salida)
model.add(Dense(parametros['num_classes'],activation='softmax'))
model.summary()
checkpoint = ModelCheckpoint('mejor_modelo.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
#%%
model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=parametros['learning_rate'],beta_1=parametros['beta_1']),metrics=['accuracy'])
# %%
history = model.fit(x_trv, y_trc,
                    batch_size=parametros['batch_size'],
                    epochs=parametros['epochs'],
                    verbose=1,
                    validation_data=(x_tv, y_tc),
                    callbacks=[checkpoint],
                    )
# %%
metricas=history.history
#print(metricas)
Ep=[p+1 for p in range(parametros['epochs'])]
print(Ep)
#Experimento.log_model("MNIST1", "mejor_modelo.hdf5")
#Experimento.end()
#graficamos las metricas contra las epocas para ver el entrenaiento 
#y detectar sobreajuste 
plt.plot(Ep,metricas['loss'],label='tr_loss',color='b')
plt.plot(Ep,metricas['val_loss'],label='val_loss',color='r')
plt.plot(Ep,metricas['accuracy'],label='tr_acurracy',color='g')
plt.plot(Ep,metricas['val_accuracy'],label='val_acurracy',color='y')
plt.legend()
plt.xlabel('epocas')
plt.ylabel('tr_loss')
plt.title('funcion de costo y accuracy')
plt.show()
# %%
