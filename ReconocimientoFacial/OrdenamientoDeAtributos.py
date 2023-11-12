#%%
# Estraeremos los atributos que consideramos importyantes para la clasificacion de imagenes de acuerdo a los atributos que ya tenemos 
# de la base de datos CelebA
from IPython.display import Image
import pandas as pd
import numpy as np
import matplotlib as plt
from pandas import Series, DataFrame
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#file_name = "C:/Users/jesus/Documents/buap/python/Redes otoño2023/img_align_celeba/Atributos2.txt"
fname = os.path.join("C:/Users/jesus/Documents/buap/python/Redes otoño2023/img_align_celeba/Atributos.txt")
with open(fname) as f:
   data = f.read()
f.close()
lines = data.split("\n")
header = lines[0].split(" ")
lines = lines[1:]
print(header)
print(lines[0])
print('linea es del tipo:',type(lines))
print(len(lines))

l=['Bald Bangs Eyeglasses Goatee Gray_Hair Mouth_Slightly_Open Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Necklace Wearing_Necktie']
l=l[0].split(' ')
er=[x:=header.index(c) for c in l]

#%%
Arreglo=[]
for n in range(len(lines)):
    Arreglo.append(lines[n].split(' '))
#%%
for n in range(len(lines)):
    Arreglo[n]=[x for x in Arreglo[n] if x !='']
    Arreglo[n]=Arreglo[n][1:]
    Arreglo[n]=[int(x) for x in Arreglo[n]]
    
a=0
for n in Arreglo:
    b=0
    for x in n:
        if x==(-1):
            Arreglo[a][b]=0
        else:
            Arreglo[a][b]=1
        b=b+1
    a=a+1
Arreglo.pop(202599)
#%%
print(er)
print(type(Arreglo[0]))
for x in range(len(Arreglo)):
    a=0
    for c in er:
        c=c-a
        Arreglo[x].pop(c)
        a=a+1
#%%
for n in Arreglo:
    if len(n)!=26:
        print('cuidato')
# %%
Arreglo=np.array(Arreglo)
#%%
for c in l:
        header.remove(c)
#%% Red neuronal para entrenamiento 
#definimos los datos de entrenamiento 
numero_base = 1

# Número de etiquetas que deseas generar
cantidad_etiquetas = 550

# Lista para almacenar las etiquetas
rutas = []

for i in range(cantidad_etiquetas):
    etiqueta = str(numero_base).zfill(6)  # Agrega ceros a la izquierda para que tenga 5 dígitos
    rutas.append(f'C:/Users/jesus/Documents/buap/python/Redes otoño2023/img_align_celeba/imagenes de prueba/{etiqueta}.jpg')
    numero_base += 1
print(len(Arreglo[:550]))
print(len(rutas))
rutas=np.array(rutas)
# %%
# Lista de etiquetas de salida
etiquetas = np.array(Arreglo[:550])
print(etiquetas)
# Tamaño al que redimensionarás las imágenes
image_size = (178, 218)

# Carga y preprocesa las imágenes
datagen = ImageDataGenerator(rescale=1.0 / 255)

# Utiliza el generador para cargar y preprocesar las imágenes
image_data = datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'image_path': rutas, 'label': Arreglo[:550]}),
    x_col='image_path',  # Columna que contiene las rutas de archivo de las imágenes
    y_col=['label'],       # Columna que contiene las etiquetas
    target_size=image_size,
    batch_size=10,  # Tamaño del lote de imágenes
    class_mode='multi_output'# Cambia a 'binary' si es una clasificación binaria
)

#%%
for data_batch, label_batch in image_data:
    # Imprime la estructura de una muestra
    print("Data batch shape:", data_batch.shape)
    print("Label batch shape:", label_batch.shape)

    # Puedes imprimir una muestra específica si lo deseas
    print("Data batch:", data_batch[0])
    print("Label batch:", label_batch[0])

    # Puedes salir del bucle después de revisar una muestra si lo deseas
    break
#%%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator #ayuda a genera cambios en las imagenes.
from tensorflow.keras.preprocessing import image
# %%
model = Sequential()
model.add(Conv2D(10, (3, 3), input_shape=(178, 218,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(10, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(20, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(10, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(16,activation='relu',kernel_regularizer=l2(0.0001)))
model.add(Dense(26))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])
model.fit(image_data,epochs=10 )
# %%
