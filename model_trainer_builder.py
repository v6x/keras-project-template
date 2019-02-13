from models.cyclegan_combined import CycleganCombined
from models.patchgan_discriminator import PatchGanDiscriminator
from models.resnet_generator import ResnetGenerator
from models.with_load_weights import WithLoadWeights, WithLoadOptimizerWeights
from trainers.cyclegan_trainer import CycleGanModelTrainer


def get_generator_model_builder(config):
    model_name = config.model.generator.model
    if model_name == 'resnet':
        return ResnetGenerator(config)
    else:
        raise ValueError(f"unknown generator model {model_name}")


def get_discriminator_model_builder(config):
    model_name = config.model.discriminator.model
    if model_name == 'patchgan':
        return PatchGanDiscriminator(config)
    else:
        raise ValueError(f"unknown discriminator model {model_name}")


# returns combined_model (for load saved model), trainer
def build_model_and_trainer(config, data_loader):
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
    else:
        raise ValueError(f"unknown model structure {model_structure}")
