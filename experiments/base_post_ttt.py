#  Copyright 2019 Gabriele Valvano
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from data_interface.interfaces.dataset_wrapper import DatasetInterfaceWrapper
from architectures.unet import UNet
from architectures.adaptor import Adaptor
from architectures.improved_discriminator import Discriminator
from idas.losses.tf_losses import weighted_cross_entropy
from idas.utils import ProgressBar
from idas.tf_utils import from_one_hot_to_rgb
import gan_losses
import config as run_config
from test_utils import test_post_adv_adaptor_ttt, test_visuals_only


class BaseExperiment(DatasetInterfaceWrapper):
    def __init__(self, run_id=None, config=None):
        """
        :param run_id: (str) used when we want to load a specific pre-trained model. Default run_id is taken from
                config_file.py
        :param config: argument from parser
        """

        self.args = run_config.define_flags() if (config is None) else config

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.CUDA_VISIBLE_DEVICE)
        self.verbose = self.args.verbose

        self.num_threads = self.args.num_threads

        # -----------------------------
        # Model hyper-parameters:
        self.lr = tf.Variable(self.args.lr, dtype=tf.float32, trainable=False, name='learning_rate')

        # -----------------------------
        # Callbacks
        # init the list of callbacks to be called and relative arguments
        self.last_val_loss = tf.Variable(1e10, dtype=tf.float32, trainable=False, name='last_val_loss')
        self.update_last_val_loss = self.last_val_loss.assign(
            tf.placeholder(tf.float32, None, name='best_val_loss_value'), name='update_last_val_loss')

        # -----------------------------
        # Other settings

        # Define global step for training e validation and counter for global epoch:
        self.g_train_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_train_step')
        self.g_valid_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_validation_step')
        self.g_test_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_test_step')
        self.g_epoch = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_epoch')

        # define their update operations
        up_value = tf.placeholder(tf.int32, None, name='update_value')
        self.update_g_train_step = self.g_train_step.assign(up_value, name='update_g_train_step')
        self.update_g_valid_step = self.g_valid_step.assign(up_value, name='update_g_valid_step')
        self.update_g_test_step = self.g_test_step.assign(up_value, name='update_g_test_step')
        self.increase_g_epoch = self.g_epoch.assign_add(1, name='increase_g_epoch')

        # training or test mode (needed for the behaviour of dropout, BN, ecc.)
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        # get gan type and losses
        gan_types = {'lsgan': gan_losses.LeastSquareGAN}
        self.gan = gan_types[self.args.gan]

        # -----------------------------
        # initialize wrapper to the data set
        super().__init__(augment=self.augment,
                         standardize=self.standardize,
                         batch_size=self.batch_size,
                         input_size=self.input_size,
                         num_threads=self.num_threads,
                         verbose=self.args.verbose)

        # -----------------------------
        # initialize placeholders for the class
        # data pipeline placeholders:
        self.global_seed = None
        self.test_init = None
        self.input_images = None
        self.output_masks = None
        # tensors of the model:
        self.optimizer = None
        self.adapted_input = None
        self.test_time_train_op = None
        self.adaptation_loss = None
        # metrics:
        self.segmentor_loss = None
        self.segmentor_dice_loss = None
        # summaries:
        self.sup_valid_images_summary_op = None
        self.sup_test_images_summary_op = None

        # -----------------------------
        # output for test interface:
        self.test_init = None
        self.valid_init = None
        self.input = None
        self.prediction = None
        self.disc_pred_fake = None
        self.soft_prediction = None
        self.ground_truth = None

        # -----------------------------
        # progress bar
        self.progress_bar = ProgressBar(update_delay=20)

    def build(self):
        """ Build the computation graph """
        if self.verbose:
            print('Building the computation graph...')
        self.get_data()
        self.define_model()
        self.define_losses()
        self.define_optimizers()
        self.define_eval_metrics()
        self.define_summaries()

    def get_data(self):
        """ Define the dataset iterators
        They will be used in define_model().
        """
        raise NotImplementedError

    def define_model(self):
        """ Define the network architecture. """

        self.input = tf.placeholder(tf.float32, [None, self.input_size[0], self.input_size[1], 1])

        # --------------
        # Adaptor  --->  I2NI
        with tf.variable_scope('Adaptor'):
            adaptor = Adaptor(n_filters=16, n_layers=3, trainable=True)
            adaptor = adaptor.build(incoming=self.input)
            self.adapted_input = adaptor.get_prediction()

        # --------------
        # Segmentor: classic UNet  --->  NI2S
        with tf.variable_scope('Segmentor'):
            unet = UNet(n_out=self.n_classes, n_filters=32, is_training=self.is_training, name='UNet')
            unet_test = unet.build(self.adapted_input)
            pred_soft = unet_test.get_prediction(softmax=True)
            prediction = unet_test.get_prediction(one_hot=True)


        with tf.variable_scope('Discriminator'):
            discriminator = Discriminator(self.is_training, n_filters=32, out_mode='scalar',
                                          instance_noise=False,
                                          label_flipping=False,
                                          one_sided_label_smoothing=False,
                                          trainable=False)
            fake_input = pred_soft[..., 1:]
            discriminator = discriminator.build(fake_input, zero_noise=True)
            self.disc_pred_fake = discriminator.get_prediction()

        # --------------
        # final prediction:
        # self.input --> it is the placeholder on top of the method
        self.prediction = prediction
        self.soft_prediction = pred_soft
        self.ground_truth = self.output_masks

    def define_losses(self):
        """
        Define loss function for each task.
        """
        # _______
        # Weighted Cross Entropy loss:
        with tf.variable_scope('WXEntropy_loss'):
            self.segmentor_loss = weighted_cross_entropy(y_pred=self.soft_prediction,
                                                         y_true=self.output_masks,
                                                         num_classes=self.n_classes)

        # define losses for supervised, unsupervised and frame prediction steps:
        w_adv = 0.1
        generator_loss = self.gan.generator_loss(self.disc_pred_fake)
        self.adaptation_loss = w_adv * generator_loss

    def define_optimizers(self):
        """
        Define training op
        using Adam Gradient Descent to minimize cost
        """
        # define lr decay
        decay = 1e-4
        self.lr = self.lr / (1. + tf.multiply(decay, tf.cast(self.g_epoch, tf.float32)))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):

            # -----------------
            # Test-time train op
            with tf.name_scope("GeneratorOptimizer"):
                optimizer = tf.train.AdamOptimizer(self.lr)
                variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Adaptor')
                gen_grads_and_vars = optimizer.compute_gradients(self.adaptation_loss, var_list=variables)
                test_time_train_op = optimizer.apply_gradients(gen_grads_and_vars, global_step=self.g_train_step)

        self.optimizer = optimizer
        self.test_time_train_op = test_time_train_op

    def define_eval_metrics(self):
        """
        Evaluate the model on the current batch
        """
        pass

    def define_summaries(self):
        """
        Create summaries to write on TensorBoard
        """
        # Scalar summaries:
        with tf.name_scope('Test'):
            test_results = list()
            test_results.append(tf.summary.image('input', self.input, max_outputs=3))
            test_results.append(tf.summary.image('prediction', from_one_hot_to_rgb(self.prediction), max_outputs=3))
            test_results.append(tf.summary.image('true', from_one_hot_to_rgb(self.ground_truth), max_outputs=3))

        self.sup_test_images_summary_op = tf.summary.merge(test_results)

    def load_best_model(self, session, saver):
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoint_dir + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print('\n\033[1;33mNo checkpoint to load under: \n{0}\033[0m\n'
                  .format(self.checkpoint_dir + '/checkpoint'))
            raise FileNotFoundError

    def test(self, sess=None):

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            adaptor_variables = {v for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="Adaptor/")}
            optimizer_variables = {v for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="GeneratorOptimizer/")}
            variables = adaptor_variables
            variables.update(optimizer_variables)
            adaptor_saver = tf.train.Saver(var_list=variables)

            segmentor_variables = {v for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="Segmentor/")}
            segmentor_variables.update({self.g_epoch})
            segmentor_saver = tf.train.Saver(var_list=segmentor_variables)

            disc_variables = {v for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="Discriminator/")}
            disc_saver = tf.train.Saver(var_list=disc_variables)

            # load weights from checkpoints:
            segm_ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoint_segmentor + '/checkpoint'))
            adaptor_saver.restore(sess, segm_ckpt.model_checkpoint_path)
            segmentor_saver.restore(sess, segm_ckpt.model_checkpoint_path)

            disc_ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoint_discriminator + '/checkpoint'))
            disc_saver.restore(sess, disc_ckpt.model_checkpoint_path)
            # opt_saver.restore(sess, ckpt.model_checkpoint_path)

            self.adaptation_threshold = 0.0
            print('\nThreshold for the adaptation loss: {0}\n'.format(self.adaptation_threshold))

            savers = [adaptor_saver, segmentor_saver, disc_saver]
            if self.args.visuals_only:
                print('\n\033[1;33m >> Test will only create visuals.\033[0m\n')
                test_visuals_only.test(self, sess, savers, segm_ckpt, n_test_steps=self.args.ttt_steps,
                                       n_images=10, test_augmentation=self.args.do_test_augmentation)
            else:
                test_post_adv_adaptor_ttt.test(self, sess, savers, segm_ckpt,
                                               n_test_steps=self.args.ttt_steps,
                                               n_images=10, test_augmentation=self.args.do_test_augmentation)
