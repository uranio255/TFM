import os

import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator, DataFrameIterator

from classes.BaseModelFactory import BaseModelFactory
from classes.DatasetManager import DatasetManager
from classes.ReportManager import ReportManager


class EnsembleModelManager:

    def __init__(self, configuration: dict, preprocessing_function: str,
                 ensemble_model_version: str):
        """
        Directorio donde se encuentra el fichero de índice que sirve de
        muestra para el conjunto de test.
        """
        self.ensemble_root_dir = configuration["ENSEMBLE_ROOT_DIR"]

        """
        Directorio donde se encuentran los modelos componentes a utilizar en 
        el modelo ensemble.
        """
        self.ensemble_models_dir = configuration["ENSEMBLE_MODELS_DIR"]

        """
        Versión concreta del modelo ensemble con la que se desea trabajar.
        """
        self.ensemble_model_version = ensemble_model_version

        # Establecemos el directorio donde se encuentran las imágenes.
        self.dir_images = configuration["IMAGES_DIR"]

        # Función de pre-procesado a aplicar a las imágenes.
        self.preprocessing_function = preprocessing_function

        # Dataframe que contiene las imágenes con las que se probará el
        # modelo ensemble una vez construido.
        self.image_dataframe = DatasetManager.get_image_dataframe(
            data_file=configuration["ENSEMBLE_CLASSES_FILE"])

        # Iteradores de imágenes. Cada uno de ellos se prepara aquí para
        # alimentar correctamente una arquitectura de red EfficientNet con el
        # tamaño de imagen adecuado.
        self.iterator_b0 = self.__get_image_iterator_for("EfficientNetB0")
        self.iterator_b1 = self.__get_image_iterator_for("EfficientNetB1")
        self.iterator_b2 = self.__get_image_iterator_for("EfficientNetB2")
        self.iterator_b3 = self.__get_image_iterator_for("EfficientNetB3")
        self.iterator_b4 = self.__get_image_iterator_for("EfficientNetB4")

        # Sub-modelos dentro de este modelo ensemble.
        self.models = []

    def __get_image_iterator_for(self, model_name: str) -> DataFrameIterator:
        return ImageDataGenerator(
            preprocessing_function=self.preprocessing_function
        ).flow_from_dataframe(
            dataframe=self.image_dataframe,
            directory=self.dir_images,
            x_col="image",
            y_col="class",
            class_mode="categorical",
            color_mode="rgb",
            shuffle=False,
            target_size=BaseModelFactory.get_image_input_size_for(
                model_name=model_name
            ),
            interpolation="nearest")

    def build_model(self) -> None:

        """
        Explora los modelos pre-entrenados ya existentes en el directorio y
        los carga.
        """
        for filename in os.listdir(self.ensemble_models_dir + "/" +
                                   self.ensemble_model_version):
            self.models.append(
                (
                    filename.rpartition('.')[0],
                    tf.keras.models.load_model(
                        filepath=self.ensemble_models_dir + "/" +
                        self.ensemble_model_version + "/" + filename)
                )
            )

    def evaluate(self) -> None:
        """
        Obtenemos la matriz de clases predichas de la red neuronal para
        cada imagen de nuestro conjunto de datos de test.
        """
        predictions_ensemble = []
        predictions_tuple = ()
        for model_tuple in self.models:
            if model_tuple[0].startswith("EfficientNetB0"):
                predictions_tuple = (
                    model_tuple[0],
                    model_tuple[1].predict(self.iterator_b0)
                )
            elif model_tuple[0].startswith("EfficientNetB1"):
                predictions_tuple = (
                    model_tuple[0],
                    model_tuple[1].predict(self.iterator_b1)
                )
            elif model_tuple[0].startswith("EfficientNetB2"):
                predictions_tuple = (
                    model_tuple[0],
                    model_tuple[1].predict(self.iterator_b2)
                )
            elif model_tuple[0].startswith("EfficientNetB3"):
                predictions_tuple = (
                    model_tuple[0],
                    model_tuple[1].predict(self.iterator_b3)
                )
            elif model_tuple[0].startswith("EfficientNetB4"):
                predictions_tuple = (
                    model_tuple[0],
                    model_tuple[1].predict(self.iterator_b4)
                )
            else:
                raise Exception("¡Uno de los modelos tiene un nombre "
                                "incorrecto!:", model_tuple[0])

            predictions_ensemble.append(predictions_tuple)

        """
        Obtenemos las clases verdaderas.
        """
        true_classes = np.argmax(
            self.image_dataframe.loc[:, 'MEL':'SCC'].to_numpy(),
            axis=1)

        """
        Mostramos el informe final de cada modelo componente por separado.
        Se utiliza el conjunto de test.
        """
        all_predictions = []
        for predictions_tuple in predictions_ensemble:
            all_predictions.append(predictions_tuple[1])
            EnsembleModelManager.final_report(
                model_name=predictions_tuple[0],
                y_true=true_classes,
                y_pred=np.argmax(predictions_tuple[1], axis=1),
                target_names=self.iterator_b0.class_indices)  # Da igual el
            # iterador

        """
        Obtenemos las clases predichas por el modelo ensamblado como media de
        las predicciones de cada modelo componente.
        """
        avg = np.mean(np.array(all_predictions), axis=0)
        all_predictions = np.argmax(avg, axis=1)

        EnsembleModelManager.final_report(
            model_name="ensamblado",
            y_true=true_classes,
            y_pred=all_predictions,
            target_names=self.iterator_b0.class_indices)  # Da igual el iterador

    @staticmethod
    def final_report(model_name: str, y_true, y_pred, target_names):
        """
        Por último mostramos el informe de clasificación final del modelo
        ensamblado junto con la matriz de confusión.
        """
        ReportManager.show_final_classification_report(
            class_report_title="Informe de clasificación del modelo " +
                               model_name,
            y_true=y_true,
            y_pred=y_pred,
            target_names=target_names,
            confusion_matrix_title="Matriz de confusión del modelo " +
                                   model_name
        )

        """
        Mostramos las métricas del modelo ensamblado.
        """
        ReportManager.show_metrics(
            title="Métricas finales del modelo " + model_name,
            y_true=y_true,
            y_pred=y_pred)
