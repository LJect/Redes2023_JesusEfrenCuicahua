import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from matplotlib import pyplot as plt
import numpy as np
import math

loss_tracker = keras.metrics.Mean(name="loss")

class Funsol(keras.Model):
    @property
    def metrics(self):
        return [loss_tracker] #igual cambia el loss_tracker

    def train_step(self, data):
        batch_size =10 #Calibra la resolucion de la ec.dif
        x = tf.random.uniform((batch_size,), minval=-1, maxval=1)
        eq = tf.math.cos(x)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            print(y_pred,'que carajos')
            loss = keras.losses.mean_squared_error(y_pred,eq)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        #actualiza metricas
        loss_tracker.update_state(loss)

        return {"loss": loss_tracker.result()}
#capa personalizada
class Polinomio(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(Polinomio,self).__init__()
        self.num_outputs = num_outputs
        
        self.kernel = self.add_weight("kernel",
                                shape=[self.num_outputs + 1]) #coeficientes entrenables

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        coeficientes = tf.unstack(self.kernel)
        resultado = tf.math.polyval(coeficientes, inputs) #evaluamos nuestras entradas para formar el polinomio con nuestros coeficientes entrenables
        return (resultado)

#vemos como se comporta nuestra capa
trans = Polinomio(3)
x = tf.random.uniform((3,), minval=-1, maxval=1)
y=2.
res=trans(y)
print(res)
#definimos nuestro modelo
inputs = keras.Input(shape=(1,))
print(len(inputs.shape))
print('hola')
x = Polinomio(3)(inputs)
model = Funsol(inputs=inputs,outputs=x)
model.summary()

model.compile(optimizer=Adam(learning_rate=0.012), metrics=['loss'])

x=tf.linspace(-1,1,100)
history = model.fit(x,epochs=100,verbose=1)

print(model.layers[1].trainable_weights,'lo')

x_testv = tf.linspace(-1,1,100)
a=model.predict(x_testv)
plt.plot(x_testv,tf.math.cos(x_testv))
plt.plot(x_testv,a)
plt.show()