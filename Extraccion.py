"""
Este script se utiliza para extraer una muestra reducida de ISIC 2019.
"""
# Importamos todos los paquetes necesarios
import os
import shutil
from shutil import copyfile

import numpy as np
import pandas as pd

# Establecemos el directorio raíz del dataset ISIC 2019
# ROOT_DIR = "../input/isic-2019"
ROOT_DIR = "."
# Directorio donde se encuentran todas las imágenes de entrada
# IMAGES_DIR = ROOT_DIR + "/ISIC_2019_Training_Input/ISIC_2019_Training_Input"
IMAGES_DIR = ROOT_DIR + '/images200'

# EXTRACTOR DE MUESTRA DE IMÁGENES PARA TRABAJAR EN LOCAL
OUTPUT_DIR = './out'

# Lectura del fichero que describe la clase a la que pertenece cada imagen
image_classes = pd.read_csv(ROOT_DIR + "/ISIC_2019_Training_GroundTruth.csv")

os.mkdir(OUTPUT_DIR)
#shutil.rmtree(OUTPUT_DIR)

df_sample = pd.DataFrame()
for columna in image_classes.loc[:, "MEL":"SCC"].columns:
    print(columna)
    df_sample_clase = image_classes.loc[image_classes[columna] == 1.0].sample(n=10)
    df_sample.append(df_sample_clase, ignore_index=True)
    for filename in df_sample_clase['image']:
        copyfile(IMAGES_DIR + '/' + filename + '.jpg', OUTPUT_DIR + '/' + filename + '.jpg')

df_sample.to_csv(OUTPUT_DIR + 'sample_ISIC_2019_Training_GroundTruth.csv')

shutil.make_archive("./all", 'zip', OUTPUT_DIR)