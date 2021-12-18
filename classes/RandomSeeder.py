class RandomSeeder:
    """
    Esta clase genera una semilla aleatoria que se utiliza principalmente
    para preparar el conjunto de datos de entrenamiento y validación durante
    el proceso de entrenamiento de un modelo cualquiera. Dicha semilla
    depende de dos factores: de la arquitectura del modelo utilizado y de un
    número natural considerado un número “de serie”.
    """
    @staticmethod
    def get_seed_for_model(efficientnet_btype: str, series: int) -> int:
        """
        Genera una semilla aleatoria.

        :param efficientnet_btype: arquitectura del modelo
        :param series: número de serie
        :return: Número entero, la semilla aleatoria.
        """
        switcher = {
            "EfficientNetB0": 0,
            "EfficientNetB1": 1,
            "EfficientNetB2": 2,
            "EfficientNetB3": 3,
            "EfficientNetB4": 4,
            "EfficientNetB5": 5,
            "EfficientNetB6": 6,
            "EfficientNetB7": 7
        }

        return series * 10 + switcher.get(efficientnet_btype)
