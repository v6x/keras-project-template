from typing import Optional

from keras import Model, Input
from keras.engine import Layer
from keras.layers import Conv2D, LeakyReLU, Activation, Conv2DTranspose, Add
from keras_contrib.layers import InstanceNormalization

from base.base_model import BaseModel
from utils.layer import DropoutPermanent


class ResnetGenerator(BaseModel):
    def define_model(self, model_name: str) -> Model:
        def conv2d(_input: Layer, filters: int, kernel_size: int, strides: int, dropout: bool,
                   activation: Optional[str], name_prefix: str) -> Layer:
            _x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                        name=f'{name_prefix}conv')(_input)
            _x = InstanceNormalization(axis=-1, epsilon=1e-05, name=f'{name_prefix}norm')(_x)
            if dropout:
                _x = DropoutPermanent(rate=0.5, name=f'{name_prefix}dropout')(_x)
            if activation == 'lrelu':
                _x = LeakyReLU(alpha=0.2, name=f'{name_prefix}{activation}')(_x)
            else:
                _x = Activation(activation, name=f'{name_prefix}{activation}')(_x)
            return _x

        def deconv2d(_input: Layer, filters: int, kernel_size: int, strides: int, dropout: bool,
                     activation: Optional[str], name_prefix: str) -> Layer:
            _x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                                 name=f'{name_prefix}deconv')(_input)
            _x = InstanceNormalization(axis=-1, epsilon=1e-05, name=f'{name_prefix}norm')(_x)
            if dropout:
                _x = DropoutPermanent(rate=0.5, name=f'{name_prefix}dropout')(_x)
            if activation == 'lrelu':
                _x = LeakyReLU(alpha=0.2, name=f'{name_prefix}{activation}')(_x)
            else:
                _x = Activation(activation, name=f'{name_prefix}{activation}')(_x)
            return _x

        def res_block(_input: Layer, filters: int, kernel_size: int, dropout: bool,
                      activation: Optional[str], name_prefix: str) -> Layer:
            _x = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', name=f'{name_prefix}conv1')(_input)
            if dropout:
                _x = DropoutPermanent(rate=0.5, name=f'{name_prefix}dropout1')(_x)
            _x = InstanceNormalization(axis=-1, epsilon=1e-05, name=f'{name_prefix}norm1')(_x)
            if activation == 'lrelu':
                _x = LeakyReLU(alpha=0.2, name=f'{name_prefix}{activation}1')(_x)
            else:
                _x = Activation(activation, name=f'{name_prefix}{activation}1')(_x)

            _x = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', name=f'{name_prefix}conv2')(_x)
            if dropout:
                _x = DropoutPermanent(rate=0.5, name=f'{name_prefix}dropout2')(_x)
            _x = InstanceNormalization(axis=-1, epsilon=1e-05, name=f'{name_prefix}norm2')(_x)
            return Add(name=f'{name_prefix}add')([_input, _x])

        # configs
        dropout = self.config.model.generator.dropout

        # resnet
        _input = Input(shape=(None, None, 3), name=f'{model_name}_input')

        # first 3 layers
        x = conv2d(_input, filters=64, kernel_size=7, strides=1, dropout=dropout, activation='relu',
                   name_prefix=f'{model_name}_conv_block1_')
        x = conv2d(x, filters=128, kernel_size=3, strides=2, dropout=dropout, activation='relu',
                   name_prefix=f'{model_name}_conv_block2_')
        x = conv2d(x, filters=256, kernel_size=3, strides=2, dropout=dropout, activation='relu',
                   name_prefix=f'{model_name}_conv_block3_')

        # 9 or 6 residual blocks depending on image size
        block_count = 9 if self.config.dataset.image_size >= 256 else 6
        for i in range(block_count):
            x = res_block(x, filters=256, kernel_size=3, dropout=dropout, activation='relu',
                          name_prefix=f'{model_name}_res_block{i + 1}_')

        # last 3 layers
        x = deconv2d(x, filters=128, kernel_size=3, strides=2, activation='relu', dropout=dropout,
                     name_prefix=f'{model_name}_deconv_block3_')
        x = deconv2d(x, filters=64, kernel_size=3, strides=2, activation='relu', dropout=dropout,
                     name_prefix=f'{model_name}_deconv_block2_')
        x = deconv2d(x, filters=3, kernel_size=7, strides=1, activation='tanh', dropout=False,
                     name_prefix=f'{model_name}_deconv_block1_')

        model = Model(inputs=_input, outputs=x, name=model_name)

        return model

    def build_model(self, **kargs) -> Model:
        raise NotImplementedError
