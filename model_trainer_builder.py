from typing import Tuple

from dotmap import DotMap
from keras import Model

from base.base_data_loader import BaseDataLoader
from base.base_model import BaseModel
from base.base_trainer import BaseTrainer
from models.cyclegan_combined import CycleganCombined
from models.dc_gan import SimpleGan
from models.dc_gan_generator import SimpleGenerator
from models.patchgan_discriminator import PatchGanDiscriminator
from models.resnet_generator import ResnetGenerator
from models.with_load_weights import WithLoadWeights, WithLoadOptimizerWeights
from trainers.ac_gan_mnist_trainer import CGanMnistTrainer
from trainers.cyclegan_trainer import CycleGanModelTrainer
from trainers.dc_gan_mnist_trainer import MnistTrainer


def get_generator_model_builder(config: DotMap) -> BaseModel:
    model_name = config.model.generator.model
    if model_name == 'resnet':
        return ResnetGenerator(config)
    elif model_name == 'generator':
        return SimpleGenerator(config)
    else:
        raise ValueError(f"unknown generator model {model_name}")


def get_discriminator_model_builder(config: DotMap) -> BaseModel:
    model_name = config.model.discriminator.model
    if model_name == 'patchgan':
        return PatchGanDiscriminator(config)
    elif model_name == 'dc_gan_discriminator':
        return SimpleDiscriminator(config)
    elif model_name == 'ac_gan_discriminator':
        return CGanDiscriminator(config)
    else:
        raise ValueError(f"unknown discriminator model {model_name}")


# returns combined_model (for load saved model), trainer
def build_model_and_trainer(config: DotMap, data_loader: BaseDataLoader) -> Tuple[Model, BaseTrainer]:
    model_structure = config.model.structure
    generator_builder = get_generator_model_builder(config)
    discriminator_builder = get_discriminator_model_builder(config)

    print('Create the model')
    if model_structure == 'cyclegan':
        g_xy = generator_builder.define_model(model_name='g_xy')
        g_yx = generator_builder.define_model(model_name='g_yx')
        d_x, parallel_d_x = WithLoadOptimizerWeights(discriminator_builder, model_name='d_x') \
            .build_model(model_name='d_x')
        d_y, parallel_d_y = WithLoadOptimizerWeights(discriminator_builder, model_name='d_y') \
            .build_model(model_name='d_y')
        combined_model, parallel_combined_model = WithLoadWeights(CycleganCombined(config), model_name='combined') \
            .build_model(g_xy=g_xy, g_yx=g_yx, d_x=d_x, d_y=d_y, model_name='combined')

        trainer = CycleGanModelTrainer(g_xy=g_xy, g_yx=g_yx,
                                       d_x=d_x, parallel_d_x=parallel_d_x,
                                       d_y=d_y, parallel_d_y=parallel_d_y,
                                       combined_model=combined_model, parallel_combined_model=parallel_combined_model,
                                       data_loader=data_loader, config=config)

        return combined_model, trainer
    elif model_structure == 'dc_gan':
        generator = generator_builder.define_model(model_name='generator')
        discriminator, parallel_discriminator = WithLoadWeights(discriminator_builder,
                                                                model_name='dc_gan_discriminator') \
            .build_model(model_name='dc_gan_discriminator')
        combined_model, parallel_combined_model = WithLoadWeights(SimpleGan(config), model_name='combined_model') \
            .build_model(generator=generator, discriminator=discriminator, model_name='combined_model')

        trainer = MnistTrainer(generator=generator,
                               discriminator=discriminator,
                               parallel_discriminator=parallel_discriminator,
                               combined_model=combined_model,
                               parallel_combined_model=parallel_combined_model,
                               data_loader=data_loader,
                               config=config)

        return combined_model, trainer
    elif model_structure == 'ac_gan':
        generator = generator_builder.define_model(model_name='generator')
        discriminator, parallel_discriminator = WithLoadWeights(discriminator_builder,
                                                                model_name='ac_gan_discriminator') \
            .build_model(model_name='ac_gan_discriminator')
        combined_model, parallel_combined_model = WithLoadWeights(CGan(config), model_name='ac_gan_combined_model') \
            .build_model(generator=generator, discriminator=discriminator, model_name='ac_gan_combined_model')

        trainer = CGanMnistTrainer(generator=generator,
                                   discriminator=discriminator,
                                   parallel_discriminator=parallel_discriminator,
                                   combined_model=combined_model,
                                   parallel_combined_model=parallel_combined_model,
                                   data_loader=data_loader,
                                   config=config)
        return combined_model, trainer
    else:
        raise ValueError(f"unknown model structure {model_structure}")
