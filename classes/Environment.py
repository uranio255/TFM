import os
import shutil
import sys

import tensorflow as tf
import pandas as pd
import sklearn as sk


class Environment:
    """
    Representa un entorno de trabajo.
    """
    def __init__(self, environment_name: str):
        """
        Crea el objeto de entorno a partir de un nombre identificativo. Se ha
        diseñado para dar soporte a tres entornos posibles:

        LOCAL: representa la máquina de desarrollo
        KAGGLE: representa la plataforma de explotación Kaggle
        GOOGLE: representa la plataforma de explotación Google Colab

        :param environment_name: nombre identificativo del entorno.
        """
        self.environment_name = environment_name

    @staticmethod
    def print_versions() -> None:
        """
        Muestra información básica del entorno
        """
        print("Versión de Python    : ", sys.version)
        print("Versión de TensorFlow: ", tf.version.VERSION)
        print("Versión de Keras     : ", tf.keras.__version__)
        print("Versión de Pandas    : ", pd.__version__)
        print("Scikit-Learn         : ", sk.__version__)
        gpu = len(tf.config.list_physical_devices('GPU')) > 0
        print("GPU                  : ",
              "disponible" if gpu else "no disponible")

    def get_config(self, reset_dirs: bool = True) -> dict:
        """
        Obtiene el diccionario con las variables de configuración del entorno.

        :type reset_dirs: Recrea/limpia los directorios de datos asociados
        a esta sesión de construcción de modelo. Si se reanuda un
        entrenamiento interrumpido este valor deberá ser False.
        :return: Diccionario de configuración.
        """
        if self.environment_name == 'LOCAL':
            # Establecemos el directorio raíz del dataset ISIC 2019
            root_dir = "./inputISIC2019"
            # Directorio donde se encuentran todas las imágenes de entrada
            images_dir = root_dir + "/sample_images"
            # Fichero donde se encuentra la clase a la que pertenece cada
            # imagen.
            classes_file = \
                root_dir + "/sample_ISIC_2019_Training_GroundTruth.csv"
            # Este es el directorio de salida, donde se grabará cualquier
            # fichero que resulte de la ejecución de este código.
            output_dir = "./output"
            # Directorio donde se grabarán puntos de salvaguarda de los
            # modelos conforme se entrenan.
            model_dir = output_dir + "/models"
            # Directorio donde se grabarán los ficheros de predicciones finales
            # con el formato adecuado.
            submission_dir = output_dir + '/submissions'
            # Directorio donde se grabarán los ficheros de clase
            # correspondientes a los distintos conjuntos de datos que se vayan
            # creando (entrenamiento, validación y test)
            datasets_dir = output_dir + '/datasets'
            # Directorio donde se podrán guardar ficheros temporales.
            temp_dir = output_dir + '/temp'

            # Establecemos el directorio raíz del dataset que se utiliza para
            # construir el modelo ensemble.
            ensemble_root_dir = './inputEnsemble/local'
            # Directorio donde se encuentran los modelos componentes.
            ensemble_models_dir = ensemble_root_dir + '/models'
            # Fichero donde se encuentra la clase a la que pertenece cada
            # imagen del conjunto de test que se utiliza durante el
            # ensamblado del modelo final.
            ensemble_classes_file = ensemble_root_dir + "/test.csv"

        elif self.environment_name == 'KAGGLE':
            # Suponemos que estamos trabajando en la plataforma "KAGGLE".

            # Establecemos el directorio raíz del dataset ISIC 2019
            root_dir = "/kaggle/input/isic-2019"
            # Directorio donde se encuentran todas las imágenes de entrada
            images_dir = \
                root_dir + "/ISIC_2019_Training_Input/ISIC_2019_Training_Input"
            # Fichero donde se encuentra la clase a la que pertenece cada
            # imagen.
            classes_file = root_dir + "/ISIC_2019_Training_GroundTruth.csv"
            # Este es el directorio de salida, donde se grabará cualquier
            # fichero que resulte de la ejecución de este código.
            output_dir = "/kaggle/working"
            # Directorio donde se grabarán puntos de salvaguarda de los
            # modelos conforme se entrenan.
            model_dir = output_dir + "/models"
            # Directorio donde se grabarán los ficheros de predicciones finales
            # con el formato adecuado.
            submission_dir = output_dir + '/submissions'
            # Directorio donde se grabarán los ficheros de clase
            # correspondientes a los distintos conjuntos de datos que se vayan
            # creando (entrenamiento, validación y test)
            datasets_dir = output_dir + '/datasets'
            # Directorio donde se podrán guardar ficheros temporales.
            temp_dir = output_dir + '/temp'

            # Establecemos el directorio raíz del dataset que se utiliza para
            # construir el modelo ensemble.
            ensemble_root_dir = '/kaggle/input/tfm-input-modelo-ensamblado' \
                                '/inputEnsemble/kaggle'
            # Directorio donde se encuentran los modelos componentes.
            ensemble_models_dir = ensemble_root_dir + '/models'
            # Fichero donde se encuentra la clase a la que pertenece cada
            # imagen del conjunto de test que se utiliza durante el
            # ensamblado del modelo final.
            ensemble_classes_file = ensemble_root_dir + "/test.csv"

        else:
            # Suponemos que estamos trabajando en la plataforma "GOOGLE COLAB".
            # Establecemos el directorio raíz del dataset ISIC 2019
            root_dir = "/content"
            # Directorio donde se encuentran todas las imágenes de entrada
            images_dir = \
                root_dir + "/ISIC_2019_Training_Input/ISIC_2019_Training_Input"
            # Fichero donde se encuentra la clase a la que pertenece cada
            # imagen.
            classes_file = root_dir + "/ISIC_2019_Training_GroundTruth.csv"
            # Este es el directorio de salida, donde se grabará cualquier
            # fichero que resulte de la ejecución de este código.
            output_dir = "/content/drive/MyDrive/Colab Notebooks/TFM"
            # Directorio donde se grabarán puntos de salvaguarda de los
            # modelos conforme se entrenan.
            model_dir = output_dir + "/models"
            # Directorio donde se grabarán los ficheros de predicciones finales
            # con el formato adecuado.
            submission_dir = output_dir + '/submissions'
            # Directorio donde se grabarán los ficheros de clase
            # correspondientes a los distintos conjuntos de datos que se vayan
            # creando (entrenamiento, validación y test)
            datasets_dir = output_dir + '/datasets'
            # Directorio donde se podrán guardar ficheros temporales.
            temp_dir = output_dir + '/temp'

            # Establecemos el directorio raíz del dataset que se utiliza para
            # construir el modelo ensemble.
            ensemble_root_dir = root_dir + '/inputEnsemble'
            # Directorio donde se encuentran los modelos componentes.
            ensemble_models_dir = ensemble_root_dir + '/models'
            # Fichero donde se encuentra la clase a la que pertenece cada
            # imagen del conjunto de test que se utiliza durante el
            # ensamblado del modelo final.
            ensemble_classes_file = ensemble_root_dir + "/test.csv"

        if reset_dirs:
            """
            Asegurar que todos los subdirectorios de salida existen, si no, 
            crearlos.
            """
            self.reset_output_dirs(
                output_dir=output_dir,
                model_dir=model_dir,
                submission_dir=submission_dir,
                datasets_dir=datasets_dir,
                temp_dir=temp_dir)

        config = {
            "ENV_NAME": self.environment_name,
            "ROOT_DIR": root_dir,
            "IMAGES_DIR": images_dir,
            "CLASSES_FILE": classes_file,
            "OUTPUT_DIR": output_dir,
            "MODEL_DIR": model_dir,
            "SUBMISSION_DIR": submission_dir,
            "DATASETS_DIR": datasets_dir,
            "TEMP_DIR": temp_dir,
            "ENSEMBLE_ROOT_DIR": ensemble_root_dir,
            "ENSEMBLE_MODELS_DIR": ensemble_models_dir,
            "ENSEMBLE_CLASSES_FILE": ensemble_classes_file
        }

        return config

    @staticmethod
    def reset_output_dirs(output_dir: str, model_dir: str,
                          submission_dir: str, datasets_dir: str,
                          temp_dir: str) -> None:

        """
        Borra el contenido del directorio de salida (output) y recrea
        todos los subdirectorios necesarios.

        :param output_dir: directorio de salida.
        :param model_dir: directorio de modelos y salvaguardas.
        :param submission_dir: directorio de submissions.
        :param datasets_dir: directorio de datasets de entrenamiento,
        validación y test.
        :param temp_dir: directorio temporal que se utiliza durante
        entrenamientos.
        """
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

        """
        Recrear subdirectorios necesarios en el directorio de output.
        """
        os.makedirs(model_dir)
        os.makedirs(submission_dir)
        os.makedirs(datasets_dir)
        os.makedirs(temp_dir)
