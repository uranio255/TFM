import tensorflow as tf

from classes.Environment import Environment
from classes.ensemble.EnsembleModelManager import EnsembleModelManager


class EnsembleModelling:
    """
    Esta clase principal se encarga de dirigir el proceso de ensamblado de un
    modelo a partir de otros modelos componentes.
    """

    def __init__(self, params):
        self.preprocessing_function = params["PREPROCESSING_FUNCTION"]
        self.ensemble_model_version = params["ENSEMBLE_MODEL_VERSION"]

    def run_on(self, environment_name):
        # Establecemos el entorno
        environment = Environment(environment_name)

        # Obtenemos la configuración adecuada del entorno especificado.
        environment_configuration = environment.get_config()

        # Muestra información del entorno (versiones de módulos disponibles).
        print("***************************************************************")
        print("* INFORMACIÓN DEL ENTORNO                                     *")
        print("***************************************************************")
        environment.print_versions()
        print()

        print("***************************************************************")
        print("* ENSAMBLADO DEL MODELO A PARTIR DE SUB-MODELOS YA ENTRENADOS *")
        print("***************************************************************")
        # Instanciamos el gestor de modelos ensamblados
        ensemble_model_manager = EnsembleModelManager(
            configuration=environment_configuration,
            ensemble_model_version=self.ensemble_model_version,
            preprocessing_function=self.preprocessing_function
        )

        # Ahora, construimos el modelo ensamblado a partir de los sub-modelos
        # ya entrenados.
        ensemble_model_manager.build_model()

        print("***************************************************************")
        print("* EVALUACIÓN FINAL                                            *")
        print("***************************************************************")
        # Ejecutar el conjunto de datos de test y obtener las predicciones
        # para cada imagen.
        ensemble_model_manager.evaluate()


# Diccionario de parámetros
parameters = {
    # Función de pre-procesado a aplicar a las imágenes.
    "PREPROCESSING_FUNCTION":
        tf.keras.applications.efficientnet.preprocess_input,
    "ENSEMBLE_MODEL_VERSION": "v1"
}

# Ejecutamos proceso de construcción y entrenamiento del modelo.
EnsembleModelling(parameters).run_on("LOCAL")
