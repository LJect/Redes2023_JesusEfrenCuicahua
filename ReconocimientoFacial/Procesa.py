import pandas as pd

# Lee el archivo de atributos en un DataFrame de Pandas
attribute_file = 'C:/Users/jesus/Documents/buap/python/Redes otoño2023/img_align_celeba/Atributos2.txt'
df_attributes = pd.read_csv(attribute_file, delim_whitespace=True)