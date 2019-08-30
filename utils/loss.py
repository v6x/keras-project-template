from keras import Model
from keras import losses
from keras.applications import VGG19, keras_modules_injection
from keras.applications.vgg19 import preprocess_input

from utils.image import denormalize_tensor


@keras_modules_injection
def preprocess_vgg_input(image_tensor, **kwargs):
    """Gets an image tensor valued between -1 and 1 and outputs an vgg preprocessed tensor
    :param image_tensor: value should be between [-1, 1]:
    :return: tensor valued between [-VGG_BGR_MEAN, 255-VGG_BGR_MEAN]
    """
    denormed_y = denormalize_tensor(image_tensor)
    processed_y = preprocess_input(denormed_y, **kwargs)

    return processed_y


def perceptual_loss(y_true, y_pred):
    vgg = VGG19(include_top=False, weights="imagenet", input_shape=(None, None, 3))
    loss_model = Model(inputs=vgg.inputs, outputs=vgg.get_layer("block5_conv4").output)
    loss_model.trainable = False
    preprocessed_y_pred = preprocess_vgg_input(y_pred)
    preprocessed_y_true = preprocess_vgg_input(y_true)
    return losses.mse(loss_model(preprocessed_y_true), loss_model(preprocessed_y_pred))
