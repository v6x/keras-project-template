from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

from base.base_model import BaseModel


class SimpleGan(BaseModel):
    def define_model(self, generator, discriminator, model_name):
        discriminator.trainable = False

        latent = Input(shape=(100,), name='noise')
        generated_image = generator(latent)
        logit = discriminator(generated_image)
        return Model(inputs=latent, outputs=logit, name=model_name)

    def build_model(self, generator, discriminator, model_name):
        combined = self.define_model(generator, discriminator, model_name)
        optimizer = Adam(self.config.model.generator.lr)
        parallel_combined = self.multi_gpu_model(combined)

        parallel_combined.compile(optimizer=optimizer, loss='binary_crossentropy')
        return combined, parallel_combined
