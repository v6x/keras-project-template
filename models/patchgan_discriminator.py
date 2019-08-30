from typing import Optional, Tuple

from keras import Model, Input
from keras.engine import Layer
from keras.layers import Conv2D, LeakyReLU, Activation
from keras.optimizers import Adam
from keras_contrib.layers import InstanceNormalization

from base.base_model import BaseModel


class PatchGanDiscriminator(BaseModel):
    def define_model(self, model_name: str) -> Model:
        def d_block(_input: Layer, filters: int, kernel_size: Tuple[int, int], strides: int, activation: Optional[str],
                    use_norm: bool, name_prefix: str) -> Layer:
            _x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                        name=f'{name_prefix}conv')(_input)
            if use_norm:
                _x = InstanceNormalization(axis=-1, epsilon=1e-05, name=f'{name_prefix}norm')(_x)
            if activation is not None:
                if activation == 'lrelu':
                    _x = LeakyReLU(alpha=0.2, name=f'{name_prefix}{activation}')(_x)
                else:
                    _x = Activation(activation, name=f'{name_prefix}{activation}')(_x)
            return _x

        dim = 64

        _input = Input(shape=(None, None, 3), name=f'{model_name}_input')

        x = d_block(_input, filters=dim * 1, kernel_size=(4, 4), strides=2, activation='lrelu', use_norm=False,
                    name_prefix=f'{model_name}_block1_')
        x = d_block(x, filters=dim * 2, kernel_size=(4, 4), strides=2, activation='lrelu', use_norm=True,
                    name_prefix=f'{model_name}_block2_')
        x = d_block(x, filters=dim * 4, kernel_size=(4, 4), strides=2, activation='lrelu', use_norm=True,
                    name_prefix=f'{model_name}_block3_')
        x = d_block(x, filters=dim * 8, kernel_size=(4, 4), strides=1, activation='lrelu', use_norm=True,
                    name_prefix=f'{model_name}_block4_')
        x = d_block(x, filters=1, kernel_size=(4, 4), strides=1, activation=None, use_norm=False,
                    name_prefix=f'{model_name}_block5_')

        # 70 x 70 Patch GAN
        # Layer     # Kernel Size   Stride  Dilation    Padding     Input Size  Output Size     Receptive Field
        # 1         4               2       1           1           256         128             4
        # 2         4               2       1           1           128         64              10
        # 3         4               2       1           1           64          32              22
        # 4         4               1       1           3           32          32              46
        # 5         4               1       1           3           32          32              70

        return Model(inputs=_input, outputs=x, name=model_name)

    def build_model(self, model_name: str) -> Tuple[Model, Model]:
        model = self.define_model(model_name)

        optimizer = Adam(lr=self.config.model.discriminator.lr, beta_1=self.config.model.discriminator.beta1,
                         clipvalue=self.config.model.discriminator.clipvalue,
                         clipnorm=self.config.model.discriminator.clipnorm)
        optimizer = self.process_optimizer(optimizer)

        parallel_model = self.multi_gpu_model(model)
        parallel_model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
        return model, parallel_model
