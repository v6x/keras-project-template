from keras import Model
from keras import losses
from keras.applications import VGG19, keras_modules_injection
from keras.applications.vgg19 import preprocess_input


@keras_modules_injection
def preprocess_vgg_input(image_tensor, **kwargs):
    """Gets an image tensor valued between -1 and 1 and outputs an vgg preprocessed tensor
    :param image_tensor: value should be between [-1, 1]:
    :return: tensor valued between [-VGG_BGR_MEAN, 255-VGG_BGR_MEAN]
    """
    denormed_y = image_tensor * 127.5 + 127.5
    processed_y = preprocess_input(denormed_y, **kwargs)

    return processed_y


def perceptual_loss(y_true, y_pred):
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=(None, None, 3))
    loss_model = Model(inputs=vgg.inputs, outputs=vgg.get_layer('block5_conv4').output)
    loss_model.trainable = False
    preprocessed_y_pred = preprocess_vgg_input(y_pred)
    preprocessed_y_true = preprocess_vgg_input(y_true)
    return losses.mse(loss_model(preprocessed_y_true), loss_model(preprocessed_y_pred))


def get_scalar_name(name: str):
    scalar_name = name
    # loss, _loss to /loss
    if 'loss' == scalar_name:
        scalar_name = '/loss'
    if ' loss' in scalar_name:
        scalar_name = scalar_name.replace(' loss', '_loss')
    if '_loss' in scalar_name:
        scalar_name = scalar_name.replace('_loss', '/loss')

    # acc, _acc to /acc
    if ' acc' in scalar_name:
        scalar_name = scalar_name.replace(' acc', '_acc')
    if '_acc' in scalar_name:
        scalar_name = scalar_name.replace('_acc', '/acc')

    return scalar_name
