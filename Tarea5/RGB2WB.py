#programa que trenaforma una imagen RGB a escala de grises
# Librerías
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Conv2D, MaxPooling2D, Layer
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Definimos parametros necesario 
#tamaño de la imagen
h,w=236,250
imputSape=(w,h,3)
# pesos para cada color en la imagen.
R=0.2126
G=0.7152
B=0.0722
#definimos los pesos de cada canal
#direccion de la imagen
dir="C:/Users/jesus/Documents/buap/python/Redes otoño2023/Redes2023_JesusEfrenCuicahua/Red_cov/DogsCats/test/cat/cat.10009.jpg"
#abrimos la imagen y la redimencionamos, para tener un tamaño igual en toda imagen que se ingrese
image=Image.open(dir)
image = image.resize((h, w))
#la convertimos en un array para poder meterla a nuestro modelo
Iarry=np.array(image)
#definimos la capa que convertira nuestras imagenes
class RGB2WB(tf.keras.layers.Layer):
    
    def call(self,input):
        print(type(input))
        salida=R*input[:,:,:,0]+G*input[:,:,:,1]+B*input[:,:,:,2] #multiplicamos cada canal por su peso correspondiente.
        return (salida)

I_input=(np.expand_dims(Iarry,axis=0)) 
#definimos el modelo
model=Sequential()
model.add(RGB2WB(input_shape=imputSape))
model.summary()
#probamos nuestro modelos
sa=model.predict(I_input)
#verificamos el tamaño de la salida
print((sa.shape))

#mostramos la imagen a color
plt.figure(1)
plt.title("Imagen a color")
plt.imshow(Iarry)
plt.axis('off')
plt.show()
#mostramos la imagen en escala de grises
plt.figure(2)
plt.title("Imagen en escala de grises")
plt.imshow(sa[0,:,:],'gray')
plt.axis('off')
plt.show()