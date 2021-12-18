class BaseModel:
    """
    Esta clase representa un modelo concreto de la familia EfficientNet.
    """
    def __init__(self):
        self.model = None
        self.model_name = None
        self.img_width = None
        self.img_height = None
        self.fine_tune_layers = None
