from contextlib import contextmanager
from collections import defaultdict
import numpy as np


__all__ = ['task', 'attrdict', 'd_of_l', 'assure_dtype_uint8',
           'merge', 'gray2rgb', 'add_border']


@contextmanager
def task(_=''):
    yield


class attrdict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def as_dict(self):
        return dict(self)

    def filt_keys(self, prefix=''):
        d = {k: v for k, v in self.items() if k.startswith(prefix)}
        return self.__class__(d)


class d_of_l(defaultdict):
    __getattr__ = dict.__getitem__

    def __init__(self, *args, **kwargs):
        super().__init__(list, *args, **kwargs)

    def as_dict(self):
        return dict(self)

    def appends(self, d):
        for key, value in d.items():
            self[key].append(value)


def assure_dtype_uint8(image):
    def raise_unknown_float_image():
        raise ValueError('Unknown float image range. Min: {}, Max: {}'.format(min_v, max_v))

    def raise_unknown_image_dtype():
        raise ValueError('Unknown image dtype: {}'.format(image.dtype))

    max_v = image.max()
    min_v = image.min()
    if image.dtype in [np.float32, np.float64]:
        if 0 <= max_v <= 1:
            if 0 <= min_v <= 1:  # [0, 1)
                min_v, max_v = 0, 1

            elif -1 <= min_v <= 0:  # Presumably [-1, 1)
                min_v, max_v = -1, 1

            else:
                raise_unknown_float_image()

        elif 0 <= max_v <= 255:
            if 0 <= min_v <= 255:  # Presumably [0, 255)
                min_v, max_v = 0, 255

            elif -256 <= min_v <= 0:  # Presumably [-256, 255)
                min_v, max_v = -256, 255

            else:
                raise_unknown_float_image()

        else:
            raise_unknown_float_image()

        return rescale(image,
                       min_from=min_v, max_from=max_v,
                       min_to=0, max_to=255,
                       dtype='uint8')

    elif image.dtype in [np.uint8]:
        return image

    else:
        raise_unknown_image_dtype()


def rescale(img, min_from=-1, max_from=1, min_to=0, max_to=255, dtype='float32'):
    len_from = max_from - min_from
    len_to = max_to - min_to
    img = (img.astype(np.float32) - min_from) * len_to / len_from + min_to
    return img.astype(dtype)


def flatten_image_list(images, show_shape) -> np.ndarray:
    """

    :param images:
    :param tuple show_shape:
    :return:
    """
    N = np.prod(show_shape)

    if isinstance(images, list):
        images = np.array(images)

    for i in range(len(images.shape)):  # find axis.
        if N == np.prod(images.shape[:i]):
            img_shape = images.shape[i:]
            new_shape = (N,) + img_shape
            return np.reshape(images, new_shape)

    else:
        raise ValueError('Cannot distinguish images. imgs shape: %s, show_shape: %s' % (images.shape, show_shape))


def merge(images, show_shape, order='row') -> np.ndarray:
    """

    :param np.ndarray images:
    :param tuple show_shape:
    :param str order:

    :return:
    """
    images = flatten_image_list(images, show_shape)
    H, W, C = images.shape[-3:]
    I, J = show_shape
    result = np.zeros((I * H, J * W, C), dtype=images.dtype)

    for k, image in enumerate(images):
        if order.lower().startswith('row'):
            i = k // J
            j = k % J
        else:
            i = k % I
            j = k // I

        target_shape = result[i * H: (i + 1) * H, j * W: (j + 1) * W].shape
        result[i * H: (i + 1) * H, j * W: (j + 1) * W] = image.reshape(target_shape)

    return result


def gray2rgb(images):
    H, W, C = images.shape[-3:]
    if C != 1:
        return images

    if images.shape[-1] != C:
        images = np.expand_dims(images, axis=-1)

    tile_shape = np.ones(len(images.shape), dtype=int)
    tile_shape[-1] = 3
    images = np.tile(images, tile_shape)
    return images


def add_border(images, color=(0, 255, 0), border=0.07):
    H, W, C = images.shape[-3:]

    if isinstance(border, float):  # if fraction
        border = int(round(min(H, W) * border))

    T = border
    images = images.copy()
    images = assure_dtype_uint8(images)
    images[:, :T, :] = color
    images[:, -T:, :] = color
    images[:, :, :T] = color
    images[:, :, -T:] = color

    return images
