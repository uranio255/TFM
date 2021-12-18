import csv
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from isic_challenge_scoring import ClassificationScore
from keras_preprocessing.image import DataFrameIterator
from sklearn.metrics import f1_score, precision_score, recall_score, \
    balanced_accuracy_score
from tensorflow.keras.callbacks import Callback

from classes import ModelManager

"""
Clase que define un callback de Keras cuyo objetivo es el de calcular 
las siguientes medidas de un conjunto de validación que se pasa como 
parámetro a su método constructor:

- Balanced accuracy
- Macro F1
- Macro precision
- Macro recall

Se han seguido las ideas de implementación de Li (2020).
"""


class ModelMetrics(Callback):
    """
    Al instanciar este callback para nuestras métricas, hemos de pasar el objeto
    iterador del que se obtendrá el conjunto de datos de validación.
    """

    def __init__(self,
                 validation_iterator: DataFrameIterator,
                 n_classes: int,
                 temp_dir: str,
                 environment_name: str,
                 model_manager: ModelManager):

        super(ModelMetrics, self).__init__()
        # Conjunto de imágenes de validación.
        self.validation_iterator = validation_iterator
        # Número de clases que el clasificador aprende a reconocer.
        self.n_classes = n_classes
        # Directorio a utilizar para ficheros temporales.
        self.temp_dir = temp_dir
        # Entorno en el que estamos trabajando.
        self.environment_name = environment_name
        """
        Referencia al objeto ModelManager que utiliza este callback durante
        el entrenamiento.
        """
        self.model_manager = model_manager

    def on_train_begin(self, logs=None):
        """
        Crea el fichero temporal que contiene las clases reales para cada
        imagen del conjunto de validación (true.csv).

        :param logs:
        :return:
        """

        """
        Obtenemos la lista de ficheros de validación. Solo los nombres, 
        ignoramos el resto de la ruta y las extensiones (.jpg).
        """
        files = []
        for file in self.validation_iterator.filenames:
            files.append(os.path.splitext(file)[0])

        """
        Obtenemos el valor real de cada clase expresado con one-hot-encoding.
        """
        true_values = tf.keras.utils.to_categorical(
            y=self.validation_iterator.labels,
            num_classes=self.n_classes)

        """
        Creamos el fichero CSV con las clases reales del conjunto de validación.
        """
        header = ['image', 'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
        lines = np.column_stack((files, true_values)).tolist()

        """
        Escribir el fichero de clases reales.
        """
        with open(file=self.temp_dir + '/' + 'true.csv',
                  mode='w',
                  encoding='UTF8',
                  newline='') as f:
            writer = csv.writer(f)

            # Escribimos la cabecera
            writer.writerow(header)

            # Escribimos las líneas de valores de clase reales
            writer.writerows(lines)

    def on_epoch_end(self, epoch, logs={}):
        """
        Función que se ejecuta al finalizar cada época de entrenamiento.
        """

        """
        Obtener las clases predichas por el modelo para cada imagen del 
        conjunto de validación. Dado que la salida de la red neuronal utiliza
        hot-encoding, debemos traducirlas a valores de clase. Para ello 
        podemos utilizar la función de argmax de Numpy.
        """
        val_predict = np.argmax(self.model.predict(self.validation_iterator),
                                axis=1)
        # Obtener las clases reales de cada imagen del conjunto de validación.
        val_targets = self.validation_iterator.labels

        """
        Calcular los valores de f1, recall y precision promediadas 
        utilizando el método "macro" tal y como se especifica en el enunciado.
        """
        val_balanced_accuracy_isic_2019 = self.get_isic_2019_balanced_accuracy()

        val_balanced_accuracy = balanced_accuracy_score(val_targets,
                                                        val_predict)
        val_macro_f1 = f1_score(val_targets, val_predict,
                                average='macro',
                                zero_division=0)
        val_macro_precision = precision_score(val_targets, val_predict,
                                              average='macro',
                                              zero_division=0)
        val_macro_recall = recall_score(val_targets, val_predict,
                                        average='macro',
                                        zero_division=0)

        """
        Actualizar el diccionario de logs para añadir las tres medidas 
        calculadas.
        """
        logs["val_balanced_accuracy_isic_2019"] = \
            val_balanced_accuracy_isic_2019
        logs["val_balanced_accuracy"] = val_balanced_accuracy
        logs["val_macro_f1"] = val_macro_f1
        logs["val_macro_precision"] = val_macro_precision
        logs["val_macro_recall"] = val_macro_recall

        print(
            "- val_balanced_accuracy (isic_2019): {} - val_balanced_accuracy "
            "(sklearn): {} - val_macro_f1: {} - val_macro_precision: {} "
            "- val_macro_recall {}".format(val_balanced_accuracy_isic_2019,
                                           val_balanced_accuracy, val_macro_f1,
                                           val_macro_precision,
                                           val_macro_recall))

        return

    def get_isic_2019_balanced_accuracy(self) -> float:
        """
        Calcula el balanced multi-class accuracy utilizando el módulo
        isic_2019_scoring.

        :return: valor del balanced multi-class accuracy definido por ISIC 2019.
        """

        """
        Graba las predicciones del conjunto de validación en un fichero 
        temporal. Si ya existía (de una época anterior) simplemente se 
        sobreescribirá.
        """
        self.model_manager.generate_submission_file(
            image_iterator=self.validation_iterator,
            directory=self.temp_dir,
            filename='predictions.csv')

        """
        Utiliza el módulo de ISIC 2019 para obtener el balanced multi-class 
        accuracy y devolver su valor.
        """
        if self.environment_name is "LOCAL":
            from isic_challenge_scoring import ClassificationMetric
            isic_2019_scoring = ClassificationScore.from_file(
                truth_file=Path(self.temp_dir + '/true.csv'),
                prediction_file=Path(self.temp_dir + '/predictions.csv'),
                target_metric=ClassificationMetric.BALANCED_ACCURACY)
        else:
            """
            La versión del módulo de ISIC 2019 disponible en Kaggle no 
            utiliza el parámetro "target_metric", por defecto utiliza BALANCED
            ACCURACY.
            """
            isic_2019_scoring = ClassificationScore.from_file(
                truth_file=Path(self.temp_dir + '/true.csv'),
                prediction_file=Path(self.temp_dir + '/predictions.csv'))
        """
        Borra el fichero de predicción grabado una vez acabado el cálculo de 
        la medida y antes de pasar a la siguiente época de entrenamiento.
        """

        return isic_2019_scoring.to_dict().get('aggregate'). \
            get('balanced_accuracy')
