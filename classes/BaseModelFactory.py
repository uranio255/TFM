from typing import Tuple

from classes.BaseModel import BaseModel
import tensorflow as tf


class BaseModelFactory:
    """
    Esta clase es capaz de instanciar un modelo cualquiera de la familia de
    EfficientNet de Keras.
    """

    """
    Esta lista contiene la información esencial para poder construir 
    adecuadamente una red neuronal de la familia de EfficientNet. Descripción de
    los valores por orden:
    - Nombre de la red.
    - Ancho en píxeles de la imagen de entrada.
    - Alto en píxeles de la imagen de entrada.
    - Número de canales de cada imagen de entrada.
    - Tupla que informa de los números de capa donde empieza cada 
    bloque convolucional (útil para fine tuning). Todas las arquitecturas de 
    EfficientNet tienen 7 bloques convolucionales.
    """
    factory_data = [
        ("EfficientNetB0", 224, 224, 3, (7, 17, 46, 75, 119, 162, 221)),
        ("EfficientNetB1", 240, 240, 3, (7, 29, 73, 117, 176, 234, 308)),
        ("EfficientNetB2", 260, 260, 3, (7, 29, 73, 117, 176, 234, 308)),
        ("EfficientNetB3", 300, 300, 3, (7, 29, 73, 117, 191, 264, 353)),
        ("EfficientNetB4", 380, 380, 3, (7, 29, 88, 147, 236, 324, 443)),
        ("EfficientNetB5", 456, 456, 3, (7, 41, 115, 189, 293, 396, 530)),
        ("EfficientNetB6", 528, 528, 3, (7, 41, 130, 219, 338, 456, 620)),
        ("EfficientNetB7", 600, 600, 3, (7, 53, 157, 261, 410, 558, 752)),
    ]

    @staticmethod
    def get_base_model(model_name: str) -> BaseModel:
        """
        Devuelve un objeto Base Model con toda la información relativa a un
        modelo base EfficientNet.

        :param model_name: nombre del modelo EfficientNet.
        :return: objeto BaseModel
        """
        base_model = BaseModel()

        var = [data for data in
               BaseModelFactory.factory_data if data[0] == model_name]

        base_model.model_name = var[0][0]
        base_model.img_width = var[0][1]
        base_model.img_height = var[0][2]

        model = getattr(tf.keras.applications, model_name)
        base_model.model = model(
            include_top=False,
            weights="imagenet",
            input_shape=(var[0][1], var[0][2], var[0][3]),
            pooling="max"
        )

        base_model.fine_tune_layers = var[0][4]

        return base_model

    @staticmethod
    def get_image_input_size_for(model_name: str) -> Tuple[int, int]:
        """
        Devuelve la anchura y altura de imagen recomendadas de un modelo
        EfficientNet dado su nombre.

        :param model_name: nombre del modelo EfficientNet.
        :return: anchura y altura, en píxeles.
        """
        var = [data for data in
               BaseModelFactory.factory_data if data[0] == model_name]

        img_width = var[0][1]
        img_height = var[0][2]

        return img_width, img_height
