import os
from typing import Tuple

import pandas as pd
import numpy as np
from keras_preprocessing.image import ImageDataGenerator, DataFrameIterator
from sklearn.model_selection import train_test_split


class DatasetManager:
    """
    Esta clase encapsula las operaciones necesarias que se pueden realizar con
    el conjunto de datos de ISIC 2019.
    """
    def __init__(self, configuration, random_seed):
        # Establecemos la semilla aleatoria a utilizar.
        self.random_seed = random_seed

        # Establecemos el directorio de imágenes a utilizar.
        self.dir_images = configuration["IMAGES_DIR"]

        # Dataframe que contiene todas las imágenes a utilizar.
        self.df_images = self.get_image_dataframe(
            data_file=configuration["CLASSES_FILE"])

        # Directorio donde se grabarán los ficheros de clase
        # correspondientes a los distintos conjuntos de datos que se vayan
        # creando (entrenamiento, validación y test)
        self.datasets_dir = configuration["DATASETS_DIR"]

        # Datasets correspondientes a los datos que se utilizan para
        # entrenamiento, validación y test.
        self.df_train = pd.DataFrame()
        self.df_validation = pd.DataFrame()
        self.df_test = pd.DataFrame()

        # Iteradores de donde se obtendrán las imágenes del dataframe de
        # entrenamiento y validación.
        self.training_iterator = None
        self.validation_iterator = None

    def create_datasets(self, n_samples_validation: int, n_samples_test: int) \
            -> Tuple[DataFrameIterator, DataFrameIterator, DataFrameIterator]:

        """
        Divide el dataset en tres conjuntos. Uno para entrenamiento, otro para
        validación y un último para test.
        IMPORTANTE: La división se hace de forma que los conjuntos
        creados contendrán la misma proporción de clases que el conjunto
        completo original.

        :param n_samples_test: número de muestras del conjunto de test.
        :param n_samples_validation: número de muestras del conjunto de
        validación.
        :return: Tres dataframes, uno que contiene los datos de
        entrenamiento, otro los de validación y el último aquellos de test.
        """

        class_col = self.df_images.iloc[:, -1]  # Columna de la clase

        """
        Primero se apartan el número de muestras especificado para un conjunto
        de test. Este conjunto será siempre el mismo ya que forzamos la semilla
        aleatoria a un valor fijo.
        """
        df_without_test, df_test, classes_without_test, classes_test = \
            train_test_split(
                    self.df_images,
                    class_col,
                    test_size=n_samples_test,
                    random_state=1000,
                    stratify=class_col)

        """
        Una vez apartado el conjunto de test, se vuelven reservan el número 
        especificado de muestras para el conjunto de validación. El resto, 
        será el conjunto de muestras para entrenamiento. Aquí el valor de la 
        semilla aleatoria dependerá del modelo que estemos entrenando.
        """
        class_col = df_without_test.iloc[:, -1]  # Columna de la clase

        df_train, df_validation, classes_train, classes_validation = \
            train_test_split(
                df_without_test,
                class_col,
                test_size=n_samples_validation,
                random_state=self.random_seed,
                stratify=class_col)

        self.df_test = df_test
        self.df_validation = df_validation
        self.df_train = df_train

        """
        Grabar los conjuntos de datos (un fichero por cada uno) en el 
        directorio dedicado a datasets. Esto resulta útil si alguna vez 
        debemos reanudar un entrenamiento que se interrumpió por alguna 
        causa.
        """
        # Crea el directorio primero si no existe.
        if not os.path.isdir(self.datasets_dir):
            os.mkdir(self.datasets_dir)

        # Columnas que se escribirán en el fichero CSV.
        columns = ['image', 'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC',
                   'SCC', 'UNK']

        # Grabamos el conjunto de test.
        df_test_copy = self.df_test.copy()
        df_test_copy['image'] = df_test_copy['image'].str[:-4]
        df_test_copy.to_csv(
            path_or_buf=self.datasets_dir + '/test.csv',
            columns=columns,
            index=False)

        # Grabamos el conjunto de validación.
        df_validation_copy = self.df_validation.copy()
        df_validation_copy['image'] = df_validation_copy['image'].str[:-4]
        df_validation_copy.to_csv(
            path_or_buf=self.datasets_dir + '/validation.csv',
            columns=columns,
            index=False)

        # Grabamos el conjunto de entrenamiento.
        df_train_copy = self.df_train.copy()
        df_train_copy['image'] = df_train_copy['image'].str[:-4]
        df_train_copy.to_csv(
            path_or_buf=self.datasets_dir + '/train.csv',
            columns=columns,
            index=False)

        return self.df_train, self.df_validation, self.df_test

    def load_datasets(self):
        """
        Carga los conjuntos de entrenamiento, validación y test previamente
        grabados a disco en formato CSV.
        """
        self.df_train = self.get_image_dataframe(self.datasets_dir +
                                                 '/train.csv')
        self.df_validation = self.get_image_dataframe(self.datasets_dir +
                                                      '/validation.csv')
        self.df_test = self.get_image_dataframe(self.datasets_dir +
                                                '/test.csv')

        return self.df_train, self.df_validation, self.df_test

    def get_image_iterators(self, preprocessing_function, batch_size: int,
                            img_width: int, img_height: int,
                            brightness_min: float, brightness_max: float) -> \
            Tuple[DataFrameIterator, DataFrameIterator]:

        """
        Este método obtiene los iteradores de imágenes adecuados para
        entrenamiento y validación.

        :param preprocessing_function: función de pre-procesado que se
         aplicarán a las imágenes tras efectuar las operaciones de aumento
         de datos. Esta función de pre-procesado depende de la red
         pre-entrenada que se utilice posteriormente.
        :param batch_size: tamaño del batch que utilizarán los iteradores de
         imágenes.
        :param img_width: anchura en píxeles que adoptarán las imágenes que
         se obtendrán con los iteradores.
        :param img_height: altura en píxeles que adoptarán las imágenes que
         se obtendrán con los iteradores.
        :param brightness_min: cota mínima de cambio de brillo aleatorio a
         utilizar durante el aumento de datos.
        :param brightness_max: cota máxima de cambio de brillo aleatorio a
         utilizar durante el aumento de datos.
        :return:
            - training_iterator: iterador que proporcionará las imágenes de
            entrenamiento. Se aplicarán técnicas de aumento de datos a estas
            imágenes.
            - validation_iterator: iterador que proporcionará las imágenes de
            validación. No se aplicará ninguna técnica de aumento de datos a
            estas imágenes.
        """

        """
        En primer lugar debemos instanciar dos objetos ImageDataGenerator. 
        Uno estará dedicado al conjunto de datos de entrenamiento y el otro a
        los datos de validación. La diferencia entre ambos es que solo el 
        generador de imágenes de entrenamiento aplicará técnicas de aumento 
        de datos.
        La única operación común que ambos efectuarán, será la de reescalar 
        el valor de los píxeles de las imágenes utilizando el parámetro 
        "rescale".
        
        Generador de datos de entrenamiento.
        
        Parámetros:
        - preprocessing_function: Utilizamos la función de preprocesamiento 
        para asegurar que cada imagen de entrada sea pre-procesada de 
        acuerdo con las características que el modelo base necesita.
        - horizontal_flip: Se aplicará la técnica de aumento de datos de 
        manera aleatoria mediante volteos horizontales de las imágenes.
        - vertical_flip: Se aplicará la técnica de aumento de datos de 
        manera aleatoria mediante volteos verticales de las imágenes.
        - brightness_range: Se aplicará la técnica de aumento de datos de 
        manera aleatoria mediante el ajuste aleatorio del brillo de las 
        imágenes.
        """
        idg_training = ImageDataGenerator(
            preprocessing_function=preprocessing_function,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=(brightness_min, brightness_max))

        """
        Generador de datos de validación. No aplicará ninguna técnica de 
        aumento de datos, solo se mantendrá el re-escalado de los valores de 
        los píxeles de las imágenes idéntico al que se aplica a las imágenes 
        de entrenamiento y el preprocesamiento adecuado para la red 
        pre-entrenada.
        """
        idg_validation = ImageDataGenerator(
            preprocessing_function=preprocessing_function)

        """
        Utilizamos el método flow_from_dataframe para obtener dos iteradores (
        generadores de Python).
        Uno servirá para obtener las imágenes del conjunto de entrenamiento y 
        el otro para obtener las imágenes del conjunto de validación/test.
        Es importante que cada iterador se obtenga del generador adecuado. 
        El iterador de datos de entrenamiento se obtiene del generador de 
        datos de entrenamiento y análogamente para el de validación.
        
        Parámetros:
        - dataframe: dataframe de donde se obtienen las imágenes de 
        entrenamiento.
        - directory: directorio donde se encuentran las imágenes.
        - x_col: indica la columna del dataframe que contiene el nombre de 
        cada fichero de imagen.
        - y_col: indica toda la serie de columnas de clase.
        - class_mode: especifica qué formato tienen los marcadores de clase.
        Utilizamos "raw" para especificar una serie de columnas numéricas que, 
        una vez unidas, expresan la clase utilizando el método de 
        one-hot-encoding.
        - target_size: todas las imágenes leídas se cambiarán al tamaño 
        establecido por estas dimensiones.
        - interpolation: método utilizado durante la operación de cambio de
        tamaño.
        - seed: fija semilla aleatoria que se utiliza durante algunas 
        transformaciones y el orden de extracción de las imágenes del dataframe.
        """
        self.training_iterator = idg_training.flow_from_dataframe(
            dataframe=self.df_train,
            directory=self.dir_images,
            x_col="image",
            y_col="class",
            class_mode="categorical",
            color_mode="rgb",
            shuffle=True,
            batch_size=batch_size,
            target_size=(img_width, img_height),
            interpolation="nearest",
            seed=self.random_seed)

        """
        En el caso de la obtención del iterador del conjunto de validación, 
        es muy importante indicar shuffle=False. Esto facilita la utilización
        posterior de funciones como predict y classification_report para 
        evaluar el modelo al garantizarse un orden bien definido de los 
        elementos devueltos.
        """
        self.validation_iterator = idg_validation.flow_from_dataframe(
            dataframe=self.df_validation,
            directory=self.dir_images,
            x_col="image",
            y_col="class",
            class_mode="categorical",
            color_mode="rgb",
            shuffle=False,
            batch_size=batch_size,
            target_size=(img_width, img_height),
            interpolation="nearest",
            seed=self.random_seed)

        # Devolvemos los iteradores de imágenes para entrenamiento y
        # validación respectivamente.
        return self.training_iterator, self.validation_iterator

    @staticmethod
    def get_image_dataframe(data_file: str):
        """
        Prepara el dataframe de datos para que cumpla las siguientes
        características:

        - Los nombres de los ficheros de imagen deben tener una extensión
        válida.
        - Debe existir un campo de etiqueta con el nombre textual de cada clase.

        :param data_file: ruta al fichero que relaciona cada fichero de
        imagen con su clase.
        :return: el dataframe una vez tratado.
        """

        """
        Lectura del fichero que describe las imágenes del conjunto de
        datos junto a la clase a la que pertenece cada una.
        """
        df_images = pd.read_csv(data_file)

        """
        Los nombres de las imágenes vienen sin extensión en el dataframe.
        Añadimos la extensión aquí.
        """
        df_images["image"] = df_images["image"] + ".jpg"

        """
        Añadimos, además, un campo de etiqueta para identificar cada
        clase de nuestro problema.
        """
        df_images["class"] = np.argmax(df_images.iloc[:, 1:].to_numpy(), axis=1)

        df_images.loc[df_images["class"] == 0, "class"] = "C0 (MEL)"
        df_images.loc[df_images["class"] == 1, "class"] = "C1 (NV)"
        df_images.loc[df_images["class"] == 2, "class"] = "C2 (BCC)"
        df_images.loc[df_images["class"] == 3, "class"] = "C3 (AK)"
        df_images.loc[df_images["class"] == 4, "class"] = "C4 (BKL)"
        df_images.loc[df_images["class"] == 5, "class"] = "C5 (DF)"
        df_images.loc[df_images["class"] == 6, "class"] = "C6 (VASC)"
        df_images.loc[df_images["class"] == 7, "class"] = "C7 (SCC)"
        df_images.loc[df_images["class"] == 8, "class"] = "C8 (UNK)"

        return df_images
