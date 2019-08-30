from typing import Optional

from keras.engine import Layer
from keras.layers import Lambda, Dropout


def named(name: str) -> Layer:
    return Lambda(lambda x: x, name=name)


class DropoutPermanent(Dropout):
    def call(self, inputs: Layer, training: Optional[bool] = None) -> Layer:
        return super().call(inputs, False)
