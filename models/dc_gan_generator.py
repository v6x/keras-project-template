from keras.layers import Dense, Activation, Input, Reshape, Conv2D, UpSampling2D, BatchNormalization
from keras.models import Model

from base.base_model import BaseModel


class SimpleGenerator(BaseModel):
    def define_model(self, model_name):
        inputs = Input(shape=(100,))
        x = Dense(7 * 7 * 128, activation='relu')(inputs)
        x = Reshape((7, 7, 128))(x)
        x = UpSampling2D(size=2)(x)

        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(32, kernel_size=3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D(size=2)(x)

        x = Conv2D(16, kernel_size=3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(1, kernel_size=3, strides=1, padding='same')(x)
        generated_image = Activation('tanh')(x)

        return Model(inputs=inputs, outputs=generated_image, name=model_name)

    def build_model(self, model_name):
        raise NotImplementedError
