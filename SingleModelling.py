import numpy as np
import tensorflow as tf
import pandas as pd

from classes.BaseModelFactory import BaseModelFactory
from classes.RandomSeeder import RandomSeeder
from classes.ReportManager import ReportManager
from classes.Environment import Environment
from classes.DatasetManager import DatasetManager
from classes.ModelManager import ModelManager


class SingleModelling:
    """
    Esta clase dirige la construcción y entrenamiento de un modelo aislado
    EfficientNet con los datos de ISIC 2019.
    """
    def __init__(self, params):
        # Parámetros de preprocesamiento necesarios.
        self.n_samples_validation = params["N_SAMPLES_VALIDATION"]
        self.n_samples_test = params["N_SAMPLES_TEST"]
        self.preprocessing_function = params["PREPROCESSING_FUNCTION"]
        self.batch_size = params["BATCH_SIZE"]
        self.base_model_name = params["BASE_MODEL_NAME"]
        self.model_series = params["MODEL_SERIES"]
        self.model_dir = None
        self.random_seed = RandomSeeder.get_seed_for_model(
            efficientnet_btype=self.base_model_name,
            series=self.model_series)
        self.brightness_min = params["BRIGHTNESS_MIN"]
        self.brightness_max = params["BRIGHTNESS_MAX"]

        # Parámetros necesarios durante la construcción del modelo.
        self.n_classes = params["N_CLASSES"]

        # Parámetros necesarios durante el entrenamiento del modelo.
        self.n_training_epochs = params["N_TRAINING_EPOCHS"]
        self.fine_tune_start_block = params["FINE_TUNE_START_BLOCK"]

        # Iteradores de entrenamiento y validación.
        self.iterator_training = None
        self.iterator_validation = None

    def run_on(self, environment_name):
        # Establecemos el entorno
        environment = Environment(environment_name)

        # Obtenemos la configuración adecuada del entorno especificado.
        environment_configuration = environment.get_config()

        """
        Establecemos el directorio donde se almacenarán las salvaguardas de
        modelos conforme se van entrenando.
        """
        self.model_dir = environment_configuration["MODEL_DIR"]

        # Muestra información del entorno (versiones de módulos disponibles).
        print("***************************************************************")
        print("* INFORMACIÓN DEL ENTORNO                                     *")
        print("***************************************************************")
        environment.print_versions()
        print()

        print("***************************************************************")
        print("* PREPARACIÓN CONJUNTOS DE DATOS (ENTRENAMIENTO Y VALIDACIÓN) *")
        print("***************************************************************")
        # Instanciamos un nuevo gestor de dataset.
        dataset_manager = DatasetManager(
            configuration=environment_configuration,
            random_seed=self.random_seed
        )

        # Dividimos el dataset en tres partes. Una para entrenamiento, otra
        # para validación y una última para test.
        df_train, df_validation, df_test = dataset_manager.create_datasets(
            n_samples_validation=self.n_samples_validation,
            n_samples_test=self.n_samples_test)

        print("Se han apartado " + str(len(df_train)) +
              " ficheros para entrenamiento.")
        print("Se han apartado " + str(len(df_validation)) +
              " ficheros para validación.")
        print("Se han apartado " + str(len(df_test)) +
              " ficheros para test.")

        # Obtenemos la definición del modelo base (pre-entrenado) a utilizar.
        base_model_factory = BaseModelFactory()
        base_model = base_model_factory.get_base_model(self.base_model_name)

        # Obtenemos iteradores de imágenes para entrenamiento y validación.
        self.iterator_training, self.iterator_validation \
            = dataset_manager.get_image_iterators(
                preprocessing_function=self.preprocessing_function,
                batch_size=self.batch_size,
                img_width=base_model.img_width,
                img_height=base_model.img_height,
                brightness_min=self.brightness_min,
                brightness_max=self.brightness_max)

        print()
        print("***************************************************************")
        print("* CONSTRUCCIÓN Y ENTRENAMIENTO DEL MODELO                     *")
        print("***************************************************************")
        print()
        print("Nombre del modelo: ", base_model.model_name)

        # Instanciamos un nuevo gestor de modelos.
        model_manager = ModelManager(
            configuration=environment_configuration,
            model_name=base_model.model_name)

        # Construimos el modelo final a partir del modelo base pre-entrenado.
        model_manager.build_model(
            base_model=base_model.model,
            n_classes=self.n_classes
        )

        # Entrenar el modelo que tiene como base la red neuronal que
        # acabamos de crear anteriormente.
        model = model_manager.train_model(
            training_iterator=self.iterator_training,
            validation_iterator=self.iterator_validation,
            n_classes=self.n_classes,
            n_epochs=self.n_training_epochs,
            n_fine_tune_layer_from=base_model.fine_tune_layers[
                self.fine_tune_start_block - 1])

        print()
        print("***************************************************************")
        print("* RESULTADOS E INFORMES                                       *")
        print("***************************************************************")

        """
        Leer los ficheros de métricas que han sido previamente grabados en
        ficheros durante el entrenamiento
        """
        history_phase_1 = pd.read_csv(
            filepath_or_buffer=self.model_dir + '/classification_layers.csv')
        history_phase_2 = pd.read_csv(
            filepath_or_buffer=self.model_dir + '/fine_tune.csv')

        # Tras el entrenamiento del modelo, mostrar informes de resultados.
        ReportManager.plot_accuracy_loss(
            history=history_phase_1,
            title=base_model.model_name + ": Accuracy y pérdida "
                                          "(fase 1 - entrenamiento capas "
                                          "clasificadoras)")
        ReportManager.plot_accuracy_loss(
            history=history_phase_2,
            title=base_model.model_name + ": Accuracy y pérdida "
                                          "(fase 2 - fine tuning)")
        ReportManager.plot_model_metrics(
            history=history_phase_1,
            title=base_model.model_name + ": Métricas sobre validación "
                                          "(fase 1 - entrenamiento capas "
                                          "clasificadoras)")
        ReportManager.plot_model_metrics(
            history=history_phase_2,
            title=base_model.model_name + ": Métricas sobre validación "
                                          "(fase 2 - fine tuning)")

        """
        Obtenemos la matriz de clases predichas de la red neuronal para cada
        imagen de nuestro conjunto de datos. Las clases son índices.
        """
        validation_predicted = model.predict(self.iterator_validation)

        """
        Dado que la salida de la red neuronal utiliza hot-encoding, debemos 
        traducirlas a valores de clase. Para ello podemos utilizar la función
        argmax de Numpy.
        """
        validation_predicted_classes = np.argmax(validation_predicted, axis=1)

        """
        Por último mostramos el informe de clasificación final junto con la
        matriz de confusión.
        """
        ReportManager.show_final_classification_report(
            class_report_title="Informe de clasificación del modelo " +
                               base_model.model_name,
            y_true=self.iterator_validation.classes,
            y_pred=validation_predicted_classes,
            target_names=self.iterator_validation.class_indices,
            confusion_matrix_title="Matriz de confusión del modelo " +
                                   base_model.model_name,
        )

        # Por último, generar un fichero con las predicciones.
        # model_manager.generate_submission_file(self.iterator_validation)
        # print("Fichero de predicciones listo")


# Diccionario de parámetros
parameters = {
    # Número de muestras del conjunto de validación
    "N_SAMPLES_VALIDATION": 2533,
    # Número de imágenes del conjunto de test
    "N_SAMPLES_TEST": 2533,
    # Función de pre-procesado a aplicar a las imágenes.
    "PREPROCESSING_FUNCTION":
        tf.keras.applications.efficientnet.preprocess_input,
    # Modelo pre-entrenado que se utilizará como base.
    "BASE_MODEL_NAME": "EfficientNetB0",
    "MODEL_SERIES": 1,
    # Escalas de brillo a utilizar durante la aplicación de aumento de datos.
    "BRIGHTNESS_MIN": 0.3,
    "BRIGHTNESS_MAX": 0.7,
    # Tamaño del batch.
    "BATCH_SIZE": 32,
    # Número máximo de épocas de entrenamiento por cada fase.
    # Fase 1: Solo se entrenan las capas clasificadoras.
    # Fase 2: Fine-tuning, solo se entrenan ciertas partes del modelo base.
    "N_TRAINING_EPOCHS": 20,
    # Número de clases que deberá predecir el clasificador.
    "N_CLASSES": 8,
    # Bloque convolucional de la arquitectura EfficientNet a partir del cual
    # se efectuará la fase de fine-tuning. Existen 7 bloques, por tanto los
    # valores posibles van de 1 a 7.
    "FINE_TUNE_START_BLOCK": 2
}

# Ejecutamos proceso de construcción y entrenamiento del modelo.
SingleModelling(parameters).run_on("LOCAL")
