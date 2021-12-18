# Fase de análisis de las imágenes
from classes.Environment import Environment
from classes.ImageAnalyzer import ImageAnalyzer


class Analysis:
    """
    Esta clase principal se encarga e efectuar una exploración de las imágenes
    que componen el conjunto de datos de ISIC 2019.
    """

    @staticmethod
    def run_on(environment_name):
        # Establecemos el entorno
        environment = Environment(environment_name)

        # Obtenemos la configuración adecuada del entorno especificado.
        configuration = environment.get_config()

        # Muestra información del entorno (versiones de módulos disponibles).
        environment.print_versions()

        # Instanciamos un nuevo analizador de imágenes.
        image_analyzer = ImageAnalyzer(configuration)

        # Exploramos y mostramos un informe de las imágenes encontradas bajo el
        # directorio indicado.
        image_analyzer.images_report()

        # Mostrar una muestra de 5 imágenes por cada clase.
        image_analyzer.show_sample_images(5)

        # Exploramos las clases y su distribución.
        image_analyzer.classes_graph()


# Ejecutamos el análisis de imágenes
Analysis().run_on("LOCAL")
