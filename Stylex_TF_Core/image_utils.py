import matplotlib.pyplot as plt
import numpy as np

import PIL.Image


def load(filename, max_size=None):
    image = PIL.Image.open(filename)

    if max_size is not None:
        factor = max_size / np.max(image.size)

        # Scale the image's height and width.
        size = np.array(image.size) * factor

        size = size.astype(int)

        image = image.resize(size, PIL.Image.LANCZOS)

    return np.float32(image)


def save(image, filename):
    image = np.clip(image, 0.0, 255.0)

    image = image.astype(np.uint8)

    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')


def plot_single(image):
    image = np.clip(image, 0.0, 255.0)
    image = image.astype(np.uint8)
    PIL.Image.fromarray(image).show()


def plot_results(content_image, style_image, mixed_image):
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))

    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    interpolation = 'sinc'

    ax = axes.flat[0]
    ax.imshow(content_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Content")

    ax = axes.flat[1]
    ax.imshow(mixed_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Mixed")

    ax = axes.flat[2]
    ax.imshow(style_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Style")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
