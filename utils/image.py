import cv2
import numpy as np
import tensorflow as tf
from PIL import ImageFont, ImageDraw, Image
from keras import backend as K, Input, Model
from keras.layers import Lambda
from scipy import stats


def read_image(filename):
    return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)


def resize_image(image, output_size):
    return cv2.resize(image, dsize=output_size, interpolation=cv2.INTER_LINEAR)


def resize_image_by_ratio(image, ratio):
    return cv2.resize(image, dsize=(0, 0), fx=ratio, fy=ratio)


def center_crop(image, crop_size):
    image_height = image.shape[0]
    image_width = image.shape[1]
    assert image_height >= crop_size
    assert image_width >= crop_size

    height_to_reduce = image_height - crop_size
    width_to_reduce = image_width - crop_size

    if height_to_reduce > 0:
        image = image[(height_to_reduce // 2):-(height_to_reduce - height_to_reduce // 2), :, :]
    if width_to_reduce > 0:
        image = image[:, (width_to_reduce // 2):-(width_to_reduce - width_to_reduce // 2), :]

    return image


def normalize_image(x):
    return x.astype(np.float32) / 127.5 - 1 if x is not None else None


def denormalize_image(x):
    return np.array(x * 127.5 + 127.5, dtype=np.uint8) if x is not None else None


# https://github.com/aiff22/DPED/blob/60cbc28605badcc0f0350791c6e2d4e5e528ad36/utils.py#L17-L30
def gauss_kernel(kernlen=21, nsig=3, channels=3):
    interval = (2 * nsig + 1.) / kernlen
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(stats.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2)
    return out_filter


kernel_var = gauss_kernel(21, 3, 3)


def blur_tensor(x):
    return tf.nn.depthwise_conv2d(x, kernel_var, [1, 1, 1, 1], padding='SAME')


def blur_image_model():
    _input = Input(shape=(None, None, 3))
    model = Model(inputs=_input, outputs=Lambda(lambda x: blur_tensor(x))(_input))
    model.trainable = False
    return model


def rgb_to_grayscale(x):
    return K.eval(tf.image.rgb_to_grayscale(x))


grayscale_kernel = np.array([[[[0.2989], [0.5870], [0.1140]]]])


def grayscale_tensor(x):
    return tf.nn.conv2d(x, grayscale_kernel, [1, 1, 1, 1], padding='SAME')


def rgb_to_grayscale_model():
    _input = Input(shape=(None, None, 3))
    model = Model(inputs=_input, outputs=Lambda(lambda x: grayscale_tensor(x))(_input))
    model.trainable = False
    return model


def textlines2image(lines: list, image_width=256, image_height=60, start_x=3, start_y=3, font_size=20):
    # if len(lines) * font_size + start_y > image_height:
    #     raise ValueError(f"At least {len(lines) * font_size + start_y} height is needed")
    while (len(lines) * font_size + start_y > image_height) and font_size > 1:
        font_size -= 1

    image = Image.new('RGB', (image_width, image_height), (255, 255, 255))
    d = ImageDraw.Draw(image)

    pos_x = start_x
    pos_y = start_y
    step_y = font_size

    for line in lines:
        if not isinstance(line, str):
            line = f'{line:04f}'

        font_size_tmp = font_size
        while (len(line) * (font_size_tmp * 0.6) + start_x) > image_width and font_size_tmp > 1:
            font_size_tmp -= 1
        fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', font_size_tmp)

        # if len(line) * (font_size * 0.6) + start_x > image_width:
        #     raise ValueError(f"At least {len(line) * (font_size * 0.6) + start_x} width is needed")
        d.text((pos_x, pos_y), line, font=fnt, fill=(0, 0, 0))
        pos_y += step_y

    # draw an end line at the very right
    d.line([(image_width - 1, 0), (image_width - 1, image_height)], fill=(0, 0, 0), width=3)

    return image
