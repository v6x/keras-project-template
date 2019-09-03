from keras.layers import Dense, Input, Flatten, Conv2D
from keras.models import Model
from keras.optimizers import SGD

from base.base_model import BaseModel


class SimpleDiscriminator(BaseModel):
    def define_model(self, model_name):
        inputs = Input(shape=(28, 28, 1), name=f'{model_name}_input')
        x = Conv2D(32, kernel_size=3, strides=1, padding='valid')(inputs)
        x = Conv2D(64, kernel_size=3, strides=1, padding='valid')(x)
        x = Conv2D(256, kernel_size=3, strides=1, padding='valid')(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        logit = Dense(1, activation='sigmoid')(x)
        return Model(inputs=inputs, outputs=logit, name=model_name)

    def build_model(self, model_name):
        model = self.define_model(model_name)
        optimizer = SGD(self.config.model.discriminator.lr)
        parallel_model = self.multi_gpu_model(model)
        parallel_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model, parallel_model
