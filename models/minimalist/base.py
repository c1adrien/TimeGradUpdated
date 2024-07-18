
import pandas as pd
import numpy as np 

class MetaBase(type):
    pass

class BaseClass(metaclass=MetaBase):
    def __init__(self):
        super().__init__()
        self.value = "BaseClass"

    def get_value(self):
        return self.value