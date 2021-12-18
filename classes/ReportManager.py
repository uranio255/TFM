"""
Esta clase reúne toda la funcionalidad para presentar gráficas y otras
visualizaciones de resultados.
"""
import itertools

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, \
    balanced_accuracy_score


class ReportManager:
    """
    La responsabilidad del gestor de informes es la de generar las gráficas e
    informes necesarios a partir de los datos que se le proporcionen.
    """

    @staticmethod
    def plot_accuracy_loss(history: DataFrame, title: str) -> None:
        """
        Método que muestra las gráficas de evolución de accuracy y pérdida
        de un modelo tanto durante el entrenamiento como la validación a partir
        del objeto de historia correspondiente obtenido al final del
        entrenamiento.

        :param history: objeto con datos de sobre la evolución del
        entrenamiento de una red neuronal.
        :param title: Título del gráfico.
        :return: Muestra el gráfico.
        """
        # Crear una figura con dos plots en cuadrícula de 1x2
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        # Visualizamos la evolución de la accuracy
        ax1.plot(history['accuracy'])
        ax1.plot(history['val_accuracy'])
        ax1.set_title('Accuracy del modelo')
        ax1.set_ylabel('accuracy')
        ax1.set_xlabel('epoch')
        ax1.legend(['entrenamiento', 'validación'], loc='lower right')
        ax1.ax = ax1

        # Visualizamos la evolución del error cometido por el modelo
        ax2.plot(history['loss'])
        ax2.plot(history['val_loss'])
        ax2.set_title('Pérdida del modelo')
        ax2.set_ylabel('loss')
        ax2.set_xlabel('epoch')
        ax2.legend(['entrenamiento', 'validación'], loc='upper right')
        ax2.ax = ax2

        fig.suptitle(title)
        plt.show()

    @staticmethod
    def plot_model_metrics(history: DataFrame, title: str) -> None:
        """
        Función que muestra las gráficas de evolución de las métricas:
        - macro average F1 score
        - macro average precision
        - macro average recall.

        :param history: objeto con datos de sobre la evolución del
        entrenamiento de una red neuronal.
        :param title: Título del gráfico.
        :return: Muestra el gráfico.
        """
        # Crear una figura con dos plots en cuadrícula de 1x2
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        # Visualizamos la evolución del error cometido por la red
        ax1.plot(history['val_balanced_accuracy'])
        ax1.plot(history['val_macro_f1'])
        ax1.set_title('Accuracy equilibrada y Macro F1')
        ax1.set_ylabel('Valor')
        ax1.set_xlabel('epoch')
        ax1.legend(['Accuracy equilibrada', 'Macro F1'], loc='upper right')
        ax1.ax = ax1

        # Visualizamos la evolución del error cometido por la red
        ax2.plot(history['val_macro_precision'])
        ax2.plot(history['val_macro_recall'])
        ax2.set_title('Macro precision y recall')
        ax2.set_ylabel('Valor')
        ax2.set_xlabel('epoch')
        ax2.legend(['Macro Precision', 'Macro Recall'], loc='upper right')
        ax2.ax = ax2

        fig.suptitle(title)
        plt.show()

    @staticmethod
    def show_final_classification_report(class_report_title: str, y_true,
                                         y_pred, target_names,
                                         confusion_matrix_title: str) \
            -> None:
        """
        Esta función muestra un informe de clasificación y la matriz de
        confusión de uno de los modelos entrenados utilizando las imágenes
        del conjunto de validación.
        """

        """
        Muestra el informe de clasificación. Este informe incluye entre otros,
        los valores de precision, recall y f1-score por cada clase. Además 
        muestra el accuracy global del modelo.
        """
        print(class_report_title)
        print()
        print(
            classification_report(
                y_true=y_true,
                y_pred=y_pred,
                target_names=target_names,
                zero_division=0)
        )

        """
        Cálculo de la matriz de confusión comparando las etiquetas reales 
        con las predichas por el modelo.
        """
        cm = confusion_matrix(y_true=y_true,
                              y_pred=y_pred)

        """
        Presentación de la matriz de confusión utilizando la función 
        definida previamente.
        El parámetro target_names se utiliza para sustituir los índices de 
        las clases por sus nombres reales y que así sea más fácil la
        identificación de las confusiones entre las clases.
        """
        ReportManager.__plot_confusion_matrix(
            cm=cm,
            normalize=False,
            target_names=list(target_names.keys()),
            title=confusion_matrix_title)

    @staticmethod
    def __plot_confusion_matrix(cm,
                                target_names,
                                title='Confusion matrix',
                                cmap=None,
                                normalize=True):
        """
        Esta función sirve para visualizar una matriz de confusión de manera más
        clara de como se obtiene directamente de utilizar la función
        confusion_matrix del paquete sklearn.
        Nota: No ha sido escrita por mí. He obtenido el código del siguiente
        enlace: https://stackoverflow.com/questions/19233771/sklearn-plot
        -confusion-matrix-with-labels

        given a sklearn confusion matrix (cm), make a nice plot

        Arguments
        ---------
        cm:           confusion matrix from sklearn.metrics.confusion_matrix

        target_names: given classification classes such as [0, 1, 2]
                      the class names, for example: ['high', 'medium', 'low']

        title:        the text to display at the top of the matrix

        cmap:         the gradient of the values displayed from
                      matplotlib.pyplot.cm
                      see
                      http://matplotlib.org/examples/color/colormaps_reference
                     .html
                      plt.get_cmap('jet') or plt.cm.Blues

        normalize:    If False, plot the raw numbers
                      If True, plot the proportions

        Usage
        -----
        plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                                  # sklearn.metrics.confusion_matrix
                              normalize    = True,                # show proportions
                              target_names = y_labels_vals,       # list of names of the classes
                              title        = best_estimator_name) # title of graph

        Citiation
        ---------
        http://scikit-learn.org/stable/auto_examples/model_selection
        /plot_confusion_matrix.html

        """

        accuracy = np.trace(cm) / np.sum(cm).astype('float')
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(16, 12))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('Etiqueta verdadera')
        plt.xlabel('Etiqueta predicha\naccuracy={:0.4f}; error={:0.4f}'.format(
            accuracy, misclass))
        plt.show()

    @staticmethod
    def show_metrics(title: str, y_true, y_pred) -> None:
        val_balanced_accuracy = balanced_accuracy_score(y_true,
                                                        y_pred)
        val_macro_f1 = f1_score(y_true, y_pred,
                                average='macro',
                                zero_division=0)
        val_macro_precision = precision_score(y_true, y_pred,
                                              average='macro',
                                              zero_division=0)
        val_macro_recall = recall_score(y_true, y_pred,
                                        average='macro',
                                        zero_division=0)

        print(title)
        print()
        print("Balanced multiclass accuracy score: ", val_balanced_accuracy)
        print("Macro F1: ", val_macro_f1)
        print("Macro Precision: ", val_macro_precision)
        print("Macro Recall: ", val_macro_recall)
