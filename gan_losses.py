import tensorflow as tf


def gradient_penalty(x_interpolated, disc_pred_interpolated, gp_weight=10.0):
    """
    Build input variables as in the example below.
    :param x_interpolated: interpolated image
    :param disc_pred_interpolated: prediction over the interpolated image
    :param gp_weight: penalty weight
    :return:

    epsilon = tf.random.uniform([self.disc_pred_real.shape[0], 1, 1, 1], 0.0, 1.0)
    x_interpolated = epsilon * x_real + (1 - epsilon) * x_fake
    model = discriminator.build(x_interpolated, reuse=True)
    disc_pred_interpolated = model.get_prediction()

    """
    grad_d_interpolated = tf.gradients(disc_pred_interpolated, [x_interpolated])[0]
    slopes = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(grad_d_interpolated), axis=[1, 2, 3]))
    penalty = tf.reduce_mean((slopes - 1.) ** 2)
    return gp_weight * penalty


class LeastSquareGAN(object):
    """ Least squares GAN losses.
    See `Least Squares Generative Adversarial Networks` (https://arxiv.org/abs/1611.04076) for more details.
    """
    def __init__(self):
        super(LeastSquareGAN, self).__init__()
        self.real_label = 1.0
        self.fake_label = -1.0

    @staticmethod
    def generator_loss(disc_pred_fake, real_label=1.0):
        loss = 0.5 * tf.reduce_mean(input_tensor=tf.math.squared_difference(disc_pred_fake, real_label))
        return loss

    @staticmethod
    def discriminator_loss(disc_pred_real, disc_pred_fake, real_label=1.0, fake_label=0.0):
        loss = 0.5 * tf.reduce_mean(input_tensor=tf.math.squared_difference(disc_pred_real, real_label)) + \
               0.5 * tf.reduce_mean(input_tensor=tf.math.squared_difference(disc_pred_fake, fake_label))
        return loss

    @staticmethod
    def discriminator_fake_loss(disc_pred_fake, fake_label=0.0):
        loss = 0.5 * tf.reduce_mean(input_tensor=tf.math.squared_difference(disc_pred_fake, fake_label))
        return loss

    @staticmethod
    def discriminator_real_loss(disc_pred_real, real_label=1.0):
        loss = 0.5 * tf.reduce_mean(input_tensor=tf.math.squared_difference(disc_pred_real, real_label))
        return loss
