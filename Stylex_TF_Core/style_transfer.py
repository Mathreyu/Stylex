import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import PIL.Image
import vgg16

from image_utils import load, plot_results, plot_single

vgg16.data_dir = 'vgg16-tfmodel/'

def mean_squared_error(a, b):


def define_content_loss_function(session, model, content_image, layer_ids):


def gram_matrix(tensor):
    shape = tensor.get_shape()
    num_channels = int(shape[3])
    matrix = tf.reshape(tensor, shape=[-1, num_channels])
    gram = tf.matmul(tf.transpose(matrix), matrix)

    return gram

def create_style_loss(session, model, style_image, layer_ids):


def create_denoise_loss(model):
    loss = tf.reduce_sum(tf.abs(model.input[:, 1:, :, :] - model.input[:, :-1, :, :])) + \
           tf.reduce_sum(tf.abs(model.input[:, :, 1:, :] - model.input[:, :, :-1, :]))

    return loss


def style_transfer(content_image, style_image,
                   content_layer_ids, style_layer_ids,
                   weight_content=1.5, weight_style=10.0,
                   weight_denoise=0.3,
                   num_iterations=120, step_size=10.0):


if __name__ == "__main__":
    style_image = load('images/style4.jpg')
    content_image = load('images/willy_wonka_new.jpg')

    style_layer_ids = [1, 2, 3, 4]
    content_layer_ids = [4]

    mixed_image = style_transfer(content_image=content_image, style_image=style_image,
                                 content_layer_ids=content_layer_ids,
                                 style_layer_ids=style_layer_ids, weight_content=1.5, weight_style=10.0,
                                 weight_denoise=0.3,
                                 num_iterations=20, step_size=10.0)

    plot_results(content_image=content_image, style_image=style_image, mixed_image=mixed_image)
