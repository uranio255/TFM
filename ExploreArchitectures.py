from classes.BaseModelFactory import BaseModelFactory
from classes.Environment import Environment
from classes.ModelManager import ModelManager

"""
Esta clase se utilizó para explorar las capas de cada una de las 
arquitecturas de los modelos EfficientNet con el objetivo de averiguar a 
partir de qué capa empieza cada uno de los siete bloques que constituyen cada
arquitectura. Este dato es necesario para desarrollar la clase BaseModelFactory.
"""

# Establecemos el entorno
environment = Environment("LOCAL")

# Obtenemos la configuración adecuada del entorno especificado.
environment_configuration = environment.get_config()

lista = [
    "EfficientNetB0",
    "EfficientNetB1",
    "EfficientNetB2",
    "EfficientNetB3",
    "EfficientNetB4",
    "EfficientNetB5",
    "EfficientNetB6",
    "EfficientNetB7"
]

for i in lista:
    base_model_factory = BaseModelFactory()
    base_model = base_model_factory.get_base_model(i)
    model_manager = ModelManager(configuration=environment_configuration,
                                 model_name=base_model.model_name)
    model_manager.build_model(
        base_model=base_model.model,
        n_classes=8
    )
    model_manager.show_trainable_layers()
