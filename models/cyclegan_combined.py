from typing import Tuple

from keras import Model, Input
from keras.optimizers import Adam

from base.base_model import BaseModel
from utils.layer import named


class CycleganCombined(BaseModel):
    def define_model(self, g_xy: Model, g_yx: Model, d_x: Model, d_y: Model, model_name: str) -> Model:
        # Note: change trainable after compilation not applied to compiled discriminator
        d_y.trainable = False
        d_x.trainable = False

        img_x = Input(shape=(None, None, 3), name='img_x')
        img_y = Input(shape=(None, None, 3), name='img_y')

        fake_y = named('fake_y')(g_xy(img_x))
        fake_x = named('fake_x')(g_yx(img_y))

        # adversarial loss: |d_y(g_xy(x)) - 1|2
        valid_y = named('valid_y')(d_y(fake_y))
        valid_x = named('valid_x')(d_x(fake_x))

        # cycle loss: |g_yx(g_xy(x)) - x|1
        recons_x = named('recons_x')(g_yx(fake_y))
        recons_y = named('recons_y')(g_xy(fake_x))

        # identity loss: |g_yx(x) - x|1
        identity_x = named('identity_x')(g_yx(img_x))
        identity_y = named('identity_y')(g_xy(img_y))

        # combined model for training generator
        return Model(inputs=[img_x, img_y],
                     outputs=[valid_x, valid_y, recons_x, recons_y, identity_x, identity_y], name=model_name)

    def build_model(self, g_xy: Model, g_yx: Model, d_x: Model, d_y: Model, model_name: str) -> Tuple[Model, Model]:
        combined = self.define_model(g_xy, g_yx, d_x, d_y, model_name)

        optimizer = Adam(lr=self.config.model.generator.lr, beta_1=self.config.model.generator.beta1,
                         clipvalue=self.config.model.generator.clipvalue, clipnorm=self.config.model.generator.clipnorm)
        optimizer = self.process_optimizer(optimizer)

        parallel_combined = self.multi_gpu_model(combined)

        recons_weight = self.config.model.generator.recons_weight
        identity_weight = self.config.model.generator.identity_weight
        parallel_combined.compile(optimizer=optimizer,
                                  loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
                                  loss_weights=[1, 1, recons_weight, recons_weight, identity_weight, identity_weight])

        return combined, parallel_combined
