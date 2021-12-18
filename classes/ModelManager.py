import csv
import os
import timeit

import tensorflow as tf
import numpy as np
from keras_preprocessing.image import DataFrameIterator
from sklearn.utils import class_weight
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import Callback, CSVLogger

from classes.ModelMetrics import ModelMetrics


class ModelManager:
    """
    Instancia modelos EfficientNet utilizando transferencia de conocimiento a
    partir de un modelo base previamente entrenado y gestiona su entrenamiento.
    """
    def __init__(self, configuration: dict, model_name: str):
        """
        Establecemos el directorio donde se almacenarán las salvaguardas de
        modelos conforme se van entrenando.
        """
        self.model_dir = configuration["MODEL_DIR"]

        """
        Establecemos el directorio donde se guardarán los ficheros de
        predicciones final en el formato adecuado.
        """
        self.submission_dir = configuration["SUBMISSION_DIR"]

        """
        Establecemos el directorio donde se podrán guardar ficheros.
        """
        self.temp_dir = configuration["TEMP_DIR"]

        """
        Establecemos el nombre del entorno donde está corriendo la aplicación.
        """
        self.environment_name = configuration["ENV_NAME"]

        """
        Partes del modelo (una vez se haya construido):
        - model_base: el modelo base pre-entrenado.
        - model_classification_layers: las capas clasificadoras finales.
        - model_complete: el modelo completo = base + capas clasificadoras.
        """
        self.model_base = None
        self.model_classification_layers = None
        self.model_complete = None

        """
        Nombre identificativo del modelo
        """
        self.model_name = model_name

    @staticmethod
    def get_class_weights(image_iterator: DataFrameIterator) -> dict:
        """
        Función que obtiene un diccionario de pesos por clase según su grado de
        aparición en el conjunto de datos referenciado por el iterador de
        imágenes pasado como parámetro. Estos pesos se utilizarán en la
        creación de los modelos posteriormente para tratar el desequilibrio
        entre las clases.
    
        :param image_iterator: iterador de imágenes.
        :return: diccionario de correspondencia entre clases y pesos para
        utilizar en un modelo de Keras pasándolo a través del parámetro
        "class_weight" del método de entrenamiento model.fit()
        """

        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(image_iterator.labels),
            y=image_iterator.labels)
        return dict(enumerate(class_weights))

    def build_model(self, base_model: Model, n_classes: int) -> None:
        """
        Este método construye el modelo que se entrenará posteriormente.
        Se utilizará la técnica de transferencia de aprendizaje: a un
        modelo base ya pre-entrenado (que se utiliza como extractor de
        características) se añaden una serie de capas clasificadoras
        completamente conectadas que harán corresponder dichas
        características a una de las clases deseadas.

        :param base_model: Modelo base pre-entrenado.
        :param model_name: Nombre identificativo del modelo completo una vez
        construido.
        :param n_classes: Número de clases posible que deberá predecir el
        modelo.
        :return: El modelo completo listo para ser entrenado.
        """

        """
        Definimos las capas que sustituirán a la última capa del modelo base.
        Partiendo de la última capa de salida del modelo base...
        """
        self.model_base = base_model
        x = self.model_base.output

        """
        ...añadimos nuestra capas de tratamiento específicas a nuestro
        problema:
        - una capa completamente conectada de 1024 neuronas con activación RELU
        - seguida de una dropout con probabilidad del 20%
        - finalmente una con activación "softmax" para clasificar a las
        clases posibles existentes en nuestro problema.
        """
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.2)(x)
        self.model_classification_layers = Dense(units=n_classes,
                                                 activation="softmax")(x)

        """
        Construimos nuestro modelo completo uniendo la red del modelo base ya 
        entrenada al principio y las capas clasificadores completamente 
        conectadas que hemos añadido justamente después.
        """
        self.model_complete = Model(inputs=self.model_base.input,
                                    outputs=self.model_classification_layers)

    def train_model(self, training_iterator: DataFrameIterator,
                    validation_iterator: DataFrameIterator, n_classes: int,
                    n_epochs: int, n_fine_tune_layer_from: int):
        """
        Método que se encarga de entrenar un modelo base previamente
        entrenado y lo ajusta (fine tune) para que se especialice mejor a las
        imágenes de entrenamiento de nuestro conjunto de datos.
        
        El entrenamiento se resume en dos fases tal y como se presenta en
        Chollet (2020):
        - Primera fase: añadimos un grupo de capas densamente conectadas que
        sirven para efectuar la clasificación en las clases posibles. Los
        pesos de estas capas se inicializan de forma aleatoria y se entrenan. El
        resto del modelo pre-entrenado se congela para que no sufra cambios
        durante esta fase.
        - Segunda fase: las últimas capas del modelo pre entrenado se
        re-entrenan conjuntamente con las capas densamente conectadas pero
        con una tasa de aprendizaje 10 veces más baja. De esta forma el
        modelo pre-entrenado se especializa más en el reconocimiento de
        nuestras imágenes.

        :param training_iterator: iterador de imágenes usadas durante el
        entrenamiento.
        :param validation_iterator: iterador de imágenes usadas durante la
        validación.
        :param n_classes: número de clases que el clasificador deberá
        aprender a reconocer.
        :param n_epochs: número máximo de épocas que durará el entrenamiento.
        :param n_fine_tune_layer_from: número de capa del modelo
        convolucional a partir de la cual se re entrenará el modelo base para
        efectuar un fine tuning de sus pesos.
        :return:
         - modelo: modelo finalmente construido y entrenado con el mejor valor
         conseguido de F1 macro average.
         - history_phase_1: métricas obtenidas durante la primera fase de
         entrenamiento.
         - history_phase_2: métricas obtenidas durante la segunda fase de
         entrenamiento.
        """

        if self.model_complete is None:
            raise Exception("¡El modelo debe construirse primero!")
        else:
            """
            Debemos congelar todos los pesos del modelo base pre-entrenado y
            dejar que solo los pesos de las capas completamente conectadas que
            hemos añadido sean los únicos que se actualizarán durante la primera
            fase de entrenamiento que realizaremos a continuación. Es importante
            hacer esto antes de compilar el modelo.
            """
            self.freeze_base_model()

            # Por último, compilamos el modelo.
            self.model_complete.compile(optimizer=Adam(learning_rate=0.001),
                                        loss="categorical_crossentropy",
                                        metrics=["accuracy"])

            """
            Efectuamos la primera fase del entrenamiento: solo las capas
            clasificadoras del modelo.
            """
            self.train_classification_layers(
                n_classes=n_classes,
                n_epochs=n_epochs,
                training_iterator=training_iterator,
                validation_iterator=validation_iterator)

            """
            Fase de fine-tuning: preparamos y compilamos el modelo 
            adecuadamente antes de comenzar esta fase del entrenamiento.
            """
            self.prepare_model_for_fine_tune(n_fine_tune_layer_from)

            """
            Efectuamos la segunda fase del entrenamiento: fine tuning.
            """
            self.fine_tune_model(
                n_classes=n_classes,
                n_epochs=n_epochs,
                n_fine_tune_layer_from=n_fine_tune_layer_from,
                training_iterator=training_iterator,
                validation_iterator=validation_iterator)

            """
            Devolver el modelo completo entrenado final y los objetos de 
            historia de ambas fases de entrenamiento.
            """
            return self.model_complete

    def freeze_base_model(self):
        """
        Congela todas las capas del modelo base para que no cambien durante
        el entrenamiento.
        """
        self.model_base.trainable = False

    def prepare_model_for_fine_tune(self, n_fine_tune_layer_from):
        """
        Se prepara el modelo para la fase de fine-tuning del entrenamiento.
        Para ello, se ponen solo las últimas capas del modelo pre-entrenado
        como entrenables durante esta fase.
        IMPORTANTE: Es necesario asegurar que todas las capas
        BatchNormalization del modelo base se encuentran congeladas y en
        modo inferencia independientemente de los bloques donde se esté
        efectuando fine tuning. En TensorFlow 2.0, congelar una capa de
        BatchNormalization hace que funcione en modo inferencia también.

        https://keras.io/getting_started/faq/
        "Starting in TensorFlow 2.0, setting bn.trainable = False will also
         force the layer to run in inference mode."
        """
        for layer in self.model_base.layers[n_fine_tune_layer_from:]:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = True
        """
        Re-compilamos el modelo para la etapa de entrenamiento de 
        fine_tuning. La diferencia esta vez, es que utilizamos una tasa de 
        aprendizaje muy baja (aquí, diez veces más baja que la utilizada 
        en la primera fase de entrenamiento).
        """
        self.model_complete.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    def fine_tune_model(self, n_classes, n_epochs, n_fine_tune_layer_from,
                        training_iterator, validation_iterator):
        """
        Efectúa la segunda etapa de entrenamiento ("fine tuning").

        :param n_classes: número de clases que el clasificador deberá
        aprender a reconocer.
        :param n_epochs: número máximo de épocas que durará el entrenamiento.
        :param n_fine_tune_layer_from: número de capa del modelo
        :param training_iterator: iterador de las imágenes de entrenamiento.
        :param validation_iterator: iterador de las imágenes de validación.
        :return: Nada.
        """
        print()
        print("Comienza la fase 2 del entrenamiento: Fine tuning.")
        print("Se ajustan las capas a partir de la número ",
              n_fine_tune_layer_from,
              " del modelo pre-entrenado.")
        print()
        print("Arquitectura del modelo base: se muestran las capas "
              "congeladas y las que no lo están durante esta fase.")
        print()
        self.show_trainable_layers()

        """
        Tomando la idea de Chollet (2019), podemos aplicar 
        pesos a las clases para corregir el desequilibrio entre ellas a 
        través del parámetro "class_weight".
        """
        start_phase = timeit.default_timer()
        history_phase_2 = self.model_complete.fit(
            training_iterator,
            steps_per_epoch=len(training_iterator),
            epochs=n_epochs,
            validation_data=validation_iterator,
            validation_steps=len(validation_iterator),
            callbacks=self.__get_callbacks(
                n_classes=n_classes,
                validation_iterator=validation_iterator,
                training_session_name='fine_tune'
            ),
            class_weight=self.get_class_weights(training_iterator)
        )
        end_phase = timeit.default_timer()
        print("Ha terminado la fase 2. Tiempo total transcurrido (segundos): ",
              end_phase - start_phase)

        return self.model_complete

    def train_classification_layers(self, n_classes, n_epochs,
                                    training_iterator, validation_iterator):
        """
        Efectúa la primera etapa de entrenamiento que se encarga de entrenar
        las capas dedicadas a la clasificación a partir de las
        características extraídas del modelo base.

        :param n_classes: número de clases que el clasificador deberá
        aprender a reconocer.
        :param n_epochs: número máximo de épocas que durará el entrenamiento.
        :param training_iterator: iterador de las imágenes de entrenamiento.
        :param validation_iterator: iterador de las imágenes de validación.
        :return: Nada.
        """

        print()
        print("Comienza la fase 1 del entrenamiento.")
        print("Se entrenan únicamente las capas clasificadoras "
              "completamente conectadas.")
        print()
        print("Arquitectura del modelo base: se muestran las capas "
              "congeladas y las que no lo están durante esta fase.")
        print()
        self.show_trainable_layers()
        """
        Tomando la idea del artículo de Chollet (2019), podemos aplicar 
        pesos a las clases para corregir el desequilibrio entre ellas a 
        través del parámetro "class_weight".
        """
        start_phase = timeit.default_timer()
        history_phase_1 = self.model_complete.fit(
            training_iterator,
            steps_per_epoch=len(training_iterator),
            epochs=n_epochs,
            validation_data=validation_iterator,
            validation_steps=len(validation_iterator),
            callbacks=self.__get_callbacks(
                n_classes=n_classes,
                validation_iterator=validation_iterator,
                training_session_name='classification_layers'),
            class_weight=self.get_class_weights(training_iterator)
        )
        end_phase = timeit.default_timer()
        print("Ha terminado la fase 1. Tiempo total transcurrido ("
              "segundos): ",
              end_phase - start_phase)

        return

    def __get_callbacks(self, n_classes, validation_iterator,
                        training_session_name) -> [Callback]:
        """
        Esta función establece los callbacks que se utilizan en cualquier
        entrenamiento.

        :param n_classes: número de clases que el clasificador deberá
        aprender a reconocer.
        :param validation_iterator: iterador de imágenes usadas durante la
        validación.
        :param training_session_name: nombre de la fase de entrenamiento.
        :return: Lista de callbacks.
        """

        callbacks = []
        """
        Definimos un callback de ModelMetrics que nos servirá para 
        calcular las siguientes medidas:
        - macro average F1 score
        - macro average recall
        - macro average precision
        """
        callbacks.append(
            ModelMetrics(validation_iterator=validation_iterator,
                         n_classes=n_classes,
                         temp_dir=self.temp_dir,
                         environment_name=self.environment_name,
                         model_manager=self))
        """
        Definimos un callback de tipo EarlyStopping que nos permitirá 
        utilizar esta técnica durante el entrenamiento.
        Se especifica la métrica a monitorizar ("val_balanced_accuracy", el
        valor de la accuracy equilibrada) y esto hará que el proceso de 
        entrenamiento se detenga cuando no se mejore dicha métrica durante
        el número de épocas indicado en el parámetro "patience".
        El parámetro "restore_best_weights" recuperará la mejor 
        configuración de pesos de la red neuronal con el mejor valor de esa 
        métrica (no tienen por qué coincidir con los pesos de la última 
        época).
        """
        callbacks.append(
            EarlyStopping(monitor="val_balanced_accuracy_isic_2019",
                          mode="max",
                          patience=5,
                          restore_best_weights=True)
        )
        """
        Definimos un callback de tipo ModelCheckpoint para que se vayan 
        guardando los pesos del modelo con mejor valor de la métrica de
        accuracy equilibrada durante el entrenamiento.
        """
        model_check_point_cb = ModelCheckpoint(
            filepath=self.model_dir + "/" + self.model_name + ".hdf5",
            monitor="val_balanced_accuracy_isic_2019",
            mode="max",
            save_best_only=True,
            save_freq="epoch",
            verbose=1)
        """
        La siguiente línea es necesaria para que ModelCheckPoint pueda
        monitorizar una métrica calculada al final de una época (como
        lo que hace la clase ModelMetrics).
        Fuente: https://stackoverflow.com/questions/58391418/
        how-to-use-custom-metric-from-a-callback-with-earlystopping-or-
        modelcheckpoint
        """
        model_check_point_cb._supports_tf_logs = False
        callbacks.append(model_check_point_cb)

        """
        CSVLogger es un callback que permite grabar las métricas del 
        entrenamiento en un fichero cada vez que finaliza una época.
        """
        csv_logger_cb = CSVLogger(
            filename=self.model_dir + '/' + training_session_name + '.csv',
            separator=",",
            append=True)
        callbacks.append(csv_logger_cb)

        """
        Salva el estado actual del entrenamiento para poder continuar más tarde
        en caso de interrupción por cualquier causa.
        """
        callbacks.append(tf.keras.callbacks.experimental.BackupAndRestore(
            backup_dir=self.model_dir))

        return callbacks

    def show_trainable_layers(self) -> None:
        """
        Muestra el listado de las capas del modelo base indicando cuáles son
        entrenables y cuáles no lo están (es decir, están "congeladas").
        :return: Nada
        """
        if self.model_complete is None:
            raise Exception("¡El modelo debe construirse primero!")
        else:
            for i, layer in enumerate(self.model_complete.layers):
                print("Capa:", str(i).rjust(3), "(", layer.name,
                      ") - Trainable:", layer.trainable)

    def load_model(self) -> None:
        """
        Carga el modelo previamente grabado en un fichero. Se asume que dicho
        fichero tiene el mismo nombre que el del modelo ("EfficientNetB0.hdf5",
        "EfficientNetB1.hdf5", etc.)
        """
        self.model_complete = tf.keras.models.load_model(
            filepath=self.model_dir + "/" + self.model_name + ".hdf5")

    def generate_submission_file(self,
                                 image_iterator: DataFrameIterator,
                                 directory: str = None,
                                 filename: str = 'submission.csv') -> None:
        """
        Genera el fichero de predicciones del modelo utilizando las imágenes
        del iterador especificado.

        :return: Nada
        """

        if self.model_complete is None:
            raise Exception("¡El modelo debe construirse primero!")
        else:

            if directory is None:
                directory = self.submission_dir

            """
            Obtenemos la matriz de clases predichas de la red neuronal para 
            cada imagen de nuestro conjunto de datos de test.
            """
            predicted = self.model_complete.predict(image_iterator)

            """
            Obtenemos la lista de ficheros de test. Solo los nombres, ignoramos 
            el resto de la ruta y las extensiones (.jpg).
            """
            files = []
            for file in image_iterator.filenames:
                files.append(os.path.splitext(file)[0])

            """
            Creamos el fichero CSV con las predicciones del mejor modelo sobre 
            los datos de test.
            """
            header = ['image', 'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC',
                      'SCC']
            predictions = np.column_stack((files, predicted)).tolist()

            """
            Escribir el fichero de predicciones.
            """
            with open(file=directory + '/' + filename,
                      mode='w',
                      encoding='UTF8',
                      newline='') as f:
                writer = csv.writer(f)

                # Escribimos la cabecera
                writer.writerow(header)

                # Escribimos las líneas de predicciones
                writer.writerows(predictions)
