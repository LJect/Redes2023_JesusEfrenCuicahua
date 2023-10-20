#%% librerias
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from matplotlib import pyplot as plt
import numpy as np
import math
#%% definimos nuestro entrenamiento
loss_tracker = keras.metrics.Mean(name="loss")
class Funsol(Sequential):
    @property
    def metrics(self):
        return [loss_tracker]

    def train_step(self,data):
        pi=3.1415
        batch_size =100
        x = tf.random.uniform((batch_size,1), minval=-1, maxval=1) #dominio de entrenamiento
        f = 3.*tf.math.sin(pi*x) #funsion 1
       #f = 1+2.*x+4.*tf.math.pow(x,2.)  #funsion 2

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = tf.math.reduce_mean(tf.math.square(y_pred-f))

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        #actualiza metricas
        loss_tracker.update_state(loss)

        return {"loss": loss_tracker.result()}
#%% Definimos nuestro modelo
model=Funsol()    
model.add(Dense(400,activation='sigmoid', input_shape=(1,)))
model.add(Dense(300,activation='sigmoid'))
model.add(Dense(200,activation='relu'))
model.add(Dense(1))

model.summary()
#%%
model.compile(optimizer=Adam(learning_rate=0.0001), metrics=['loss'])
x=tf.linspace(-1,1,100)
#%% Entrenamos el modelo 
history = model.fit(x,epochs=4000,verbose=0)
print(history.history.keys())
#%%
#graficamos la funcion de perdida
#plt.plot(history.history["loss"])
a=model.predict(x)
#%%
# Comparamos la prediccion del modelo con la fucion exacta.
plt.plot(x,a,label="aprox")
plt.plot(x, 3.*tf.math.sin(3.1415*x), label="exact")
#plt.plot(x,1+2.*x+4.*tf.math.pow(x,2.),label="exact")
plt.legend()
plt.show()
# %%
