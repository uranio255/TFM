import collections

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class ImageAnalyzer:
    """
    Esta clase efectúa un recorrido exhaustivo de todas las imágenes que
    conforman el conjunto de datos ISIC 2019, cargándolas una a una y
    examinando sus propiedades.
    """
    def __init__(self, configuration):
        # Establecemos la semilla aleatoria a utilizar.
        self.random_seed = 1000

        # Establecemos el directorio de imágenes a utilizar.
        self.dir_images = configuration["IMAGES_DIR"]

        # Lectura del fichero que describe la clase a la que pertenece cada
        # imagen.
        self.df_images = pd.read_csv(configuration["CLASSES_FILE"])

    def images_report(self):
        """
        Esta función explora y ofrece un informe de las imágenes que se
        encuentran por debajo de un directorio raíz especificado. El informe
        contiene los siguientes elementos:

        - Rango dinámico (valores máximo y mínimo encontrados en sus matrices de
        canales). Cada imagen se obtiene como un array de Numpy con lo que
        podemos utilizar las funciones min() y max() para obtener la respuesta.
        - Tamaño de la imagen: altura x anchura.
        - Número de canales.
        """

        # Recorremos el listado de imágenes obtenido y vamos obteniendo
        # distintas métricas de cada una de ellas.
        range_min = 9999
        range_max = -1
        dimensions = {}
        channels = []
        for i in self.df_images.index:
            # Lectura de la imagen
            image = cv2.imread(
                self.dir_images + '/' + self.df_images["image"][i] + '.jpg')

            # Actualización del valor mínimo encontrado para el rango dinámico
            # global.
            if image.min() < range_min:
                range_min = image.min()

            # Actualización del valor máximo encontrado para el rango dinámico
            # global.
            if image.max() > range_max:
                range_max = image.max()

            # Actualización de la lista de dimensiones de imagen encontradas
            size_pair = (image.shape[0], image.shape[1])
            if size_pair in dimensions:
                classes_dict = dimensions.get(size_pair)
                class_key = self.df_images.columns[
                    np.argmax(self.df_images.loc[i][2:]) + 2]
                if class_key in classes_dict:
                    classes_dict.update(
                        {class_key: classes_dict.get(class_key) + 1})
                else:
                    classes_dict[class_key] = 1
                dimensions.update({size_pair: classes_dict})
            else:
                dimensions[size_pair] = {
                    self.df_images.columns[
                        np.argmax(self.df_images.loc[i][2:]) + 2]: 1
                }

            # Actualización del número de canales encontrado
            channels.append(image.shape[2])

        # Agrupamos y contamos las imágenes que tenemos por número de canales.
        count_channels = collections.Counter(channels)

        # Mostramos el rango dinámico encontrado para el dataset completo.
        print("Rango dinámico: {}-{}".format(range_min, range_max))

        # Mostramos cuántas imágenes tenemos por número de canales.
        print("Número de canales:")
        for j in count_channels.keys():
            print("  - {} -> {} imágenes ({})".format(
                str(j).rjust(7),
                count_channels.get(j),
                str("{:.2f}%".format(
                    count_channels.get(j) / len(self.df_images) * 100))
            ).rjust(6))

        # Mostramos cuántas imágenes tenemos por cada tamaño encontrado.
        print("Dimensiones:")
        for j in dimensions.keys():
            total_images = sum(dimensions.get(j).values())
            for k in dimensions.get(j):
                dimensions.get(j)[k] = "{:.2f}%".format(
                    dimensions.get(j)[k] * 100 / total_images)

            print("  - {}x{} -> {} imágenes ({}) (clases: {})".format(
                str(j[0]).rjust(4),
                str(j[1]).rjust(4),
                str(total_images).rjust(5),
                str("{:.2f}%".format(
                    total_images / len(self.df_images) * 100)).rjust(7),
                dimensions.get(j)
            ))

        # Finalmente mostramos el total de imágenes del conjunto de datos.
        print("Total imágenes del conjunto de entrenamiento:",
              len(self.df_images),
              "(100.00%)")

    def show_sample_images(self, n_images: int) -> None:
        """
        Cargamos y mostramos las primeras imágenes de cada clase usando OpenCV

        :param n_images: número de imágenes de muestra a mostrar.
        """

        for the_class in self.df_images.loc[:, "MEL":"SCC"].columns:
            print("Muestra de la clase " + the_class + ":")
            df_sample_class = \
                self.df_images.loc[self.df_images[the_class] == 1.0].sample(
                    n=n_images, random_state=self.random_seed)

            sample_images = []
            for i in df_sample_class.index:
                sample_images.append(cv2.imread(self.dir_images + "/" +
                                                df_sample_class["image"][i] +
                                                '.jpg'))

            plt.figure(figsize=(16, 10))
            for i in range(1, n_images + 1):
                plt.subplot(1, 5, i)
                plt.grid(False)
                plt.imshow(sample_images[i - 1])
            plt.show()

    def classes_graph(self) -> None:
        """
        Esta función produce un gráfico donde se puede ver la representación de
        cada clase en el conjunto de datos.
        """

        # Mostrar un extracto del contenido del fichero leído
        print(self.df_images)

        # Mostramos un gráfico con la distribución de clases del conjunto de
        # datos,
        # para ello definimos un eje Y con el número de imágenes por clase.
        y = self.df_images.loc[:, "MEL":"UNK"].sum()

        # El eje X con cada una de las clases de imágenes.
        x = self.df_images.columns[1:]

        # Ordenar los datos para apreciar mejor los posibles desequilibrios.
        y, x = zip(*sorted(zip(y, x)))

        # Gráfico que muestra los dos ejes definidos.
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.bar(x, y)
        ax.set_ylabel("Número de imágenes")
        ax.set_xlabel("Clase")
        patches = ax.patches
        labels = [f"{i}" for i in y]
        for rect, label in zip(patches, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height, label,
                    ha="center",
                    va="bottom")
        plt.show()
