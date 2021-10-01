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
import time
from idas import utils
import tensorflow as tf
from data_interface.interfaces.dataset_wrapper import DatasetInterfaceWrapper
from idas.callbacks import callbacks as tf_callbacks
from idas.callbacks.routine_callback import RoutineCallback
from idas.callbacks.early_stopping_callback import EarlyStoppingCallback, EarlyStoppingException, NeedForTestException
from architectures.unet import UNet
from architectures.adaptor import Adaptor
from architectures.improved_discriminator import Discriminator
from idas.metrics.tf_metrics import dice_coe
from idas.losses.tf_losses import weighted_cross_entropy
from tensorflow.core.framework import summary_pb2
from idas.utils import ProgressBar
import random
from idas.tf_utils import from_one_hot_to_rgb
import gan_losses
import config as run_config
from utils import tf_add_noise_boxes, tf_gaussian_noise_layer


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
        self.callbacks = []
        self.callbacks_kwargs = {'history_log_dir': self.history_log_dir}
        self.callbacks.append(RoutineCallback())  # routine callback always runs
        # Early stopping callback:
        self.callbacks_kwargs['es_loss'] = None
        self.last_val_loss = tf.Variable(1e10, dtype=tf.float32, trainable=False, name='last_val_loss')
        self.update_last_val_loss = self.last_val_loss.assign(
            tf.placeholder(tf.float32, None, name='best_val_loss_value'), name='update_last_val_loss')
        self.callbacks_kwargs['test_on_minimum'] = True
        self.callbacks.append(EarlyStoppingCallback(min_delta=1e-5, patience=2000))

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

        # lr decay
        # self.decay_lr = self.lr.assign(tf.multiply(self.lr, 1.0), name='decay_lr')
        # self.update_lr = self.lr.assign(
        #     cyclic_learning_rate(self.g_epoch, step_size=20,
        #                          learning_rate=self.args.lr // 10, max_lr=self.args.lr,
        #                          mode='triangular', gamma=.997), name='update_lr')

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
        self.sup_train_init = None
        self.sup_valid_init = None
        self.sup_test_init = None
        self.sup_input_data = None
        self.sup_output_mask = None
        self.unsup_train_init = None
        self.unsup_valid_init = None
        self.unpaired_images = None
        self.disc_train_init = None
        self.disc_valid_init = None
        self.unpaired_masks = None
        # tensors of the model:
        self.optimizer = None
        self.adapted_input = None
        self.delta_input_sup = None
        self.sup_pred_mask_soft = None
        self.sup_pred_mask_oh = None
        self.disc_pred_mask_soft = None
        self.disc_pred_mask_oh = None
        self.disc_pred_real = None
        self.disc_pred_fake = None
        self.disc_pred_synth_fake = None
        self.disc_pred_interpolated = None
        self.disc_pred_replay_interpolated = None
        self.disc_pred_replay = None
        self.replay_mask = None
        self.test_pred_fake = None
        self.replay_mask_buffer = None
        # metrics:
        self.xentropy_loss = None
        self.sup_loss = None
        self.disc_loss_real = None
        self.disc_loss_fake = None
        self.disc_loss_synth_fake = None
        self.real_fake_interpolated = None
        self.real_fake_replay_interpolated = None
        self.adv_disc_loss = None
        self.adv_gen_loss = None
        self.generator_loss = None
        self.discriminator_loss = None
        self.discriminator_loss_replay = None
        self.gradient_penalty = None
        self.gradient_penalty_replay = None
        self.adaptation_loss = None
        self.dice = None
        self.dice_sup = None
        self.dice_sup_loss = None
        self.train_op = None
        self.train_op_replay = None
        # summaries:
        self.sup_train_scalar_summary_op = None
        self.sup_valid_scalar_summary_op = None
        self.sup_valid_images_summary_op = None
        self.sup_test_images_summary_op = None
        self.all_train_scalar_summary_op = None
        self.weights_summary = None

        # -----------------------------
        # output for test interface:
        self.test_init = None
        self.input = None
        self.prediction = None
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

        # --------------
        # Adaptor  --->  I2NI
        with tf.variable_scope('Adaptor'):
            # -----------------
            # SUPERVISED BRANCH
            adaptor = Adaptor(n_filters=16, n_layers=3, trainable=True)
            adaptor = adaptor.build(incoming=self.sup_input_data)
            adapted_sup_input = adaptor.get_prediction()

            # ------------------
            # ADVERSARIAL BRANCH
            disc_adaptor = adaptor.build(incoming=self.unpaired_images, reuse=True)
            adapted_disc_input = disc_adaptor.get_prediction()

        # - - - - - - -
        # Segmentor: classic UNet  --->  NI2S
        with tf.variable_scope('Segmentor'):
            norm_type = self.args.normalization_type
            assert norm_type == 'BN'
            unet = UNet(n_out=self.n_classes, n_filters=32, is_training=self.is_training, name='UNet')

            # -----------------
            # SUPERVISED BRANCH
            unet_sup = unet.build(adapted_sup_input)
            self.sup_pred_mask_soft = unet_sup.get_prediction(softmax=True)
            self.sup_pred_mask_oh = unet_sup.get_prediction(one_hot=True)

            # ------------------
            # ADVERSARIAL BRANCH
            unet_disc = unet.build(adapted_disc_input, reuse=True)
            self.disc_pred_mask_soft = unet_disc.get_prediction(softmax=True)
            self.disc_pred_mask_oh = unet_disc.get_prediction(one_hot=True)

        # - - - - - - -
        # Build Mask Discriminator
        with tf.variable_scope('Discriminator'):
            discriminator = Discriminator(self.is_training, n_filters=32, out_mode='scalar',
                                          instance_noise=self.args.instance_noise,
                                          label_flipping=self.args.label_flipping,
                                          one_sided_label_smoothing=self.args.one_sided_label_smoothing)

            fake_input = self.disc_pred_mask_soft[..., 1:]
            model_fake = discriminator.build(fake_input)
            self.disc_pred_fake = model_fake.get_prediction()

            real_input = self.unpaired_masks[..., 1:]
            model_real = discriminator.build(real_input, reuse=True)
            self.disc_pred_real = model_real.get_prediction()

            # ------------------
            # Synthetic fakes
            synthetic_fake_mask = tf_add_noise_boxes(self.unpaired_masks, n_classes=self.n_classes, n_boxes=10,
                                                     image_size=self.input_size,
                                                     mask_type=['random', 'jigsaw', 'zeros'],
                                                     probability={'random': 0.9, 'jigsaw': 0.5, 'zeros': 0.5})
            if self.args.instance_noise:
                synthetic_fake_mask = tf_gaussian_noise_layer(synthetic_fake_mask, mean=0.0, std=0.2)

            synthetic_fake_input = synthetic_fake_mask[..., 1:]
            model_fake = discriminator.build(synthetic_fake_input, reuse=True, zero_noise=True)
            self.disc_pred_synth_fake = model_fake.get_prediction()

            # Grad. penalty: build interpolated image:
            epsilon = tf.random.uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
            x_interpolated = epsilon * self.unpaired_masks + (1 - epsilon) * synthetic_fake_mask
            self.real_fake_interpolated = x_interpolated[..., 1:]
            model_interp = discriminator.build(self.real_fake_interpolated, reuse=True, zero_noise=True)
            self.disc_pred_interpolated = model_interp.get_prediction()

            # ------------------
            # Experience Replay
            shape = [None, self.input_size[0], self.input_size[1], self.n_classes]
            self.replay_mask = tf.placeholder(tf.float32, shape)
            synthetic_fake_replay = tf_add_noise_boxes(self.replay_mask, n_classes=self.n_classes, n_boxes=10,
                                                       image_size=self.input_size,
                                                       mask_type=['random', 'jigsaw', 'zeros'],
                                                       probability={'random': 0.5, 'jigsaw': 0.1, 'zeros': 0.1})
            synthetic_fake_input = synthetic_fake_replay[..., 1:]
            model_replay = discriminator.build(synthetic_fake_input, reuse=True, zero_noise=True)
            self.disc_pred_replay = model_replay.get_prediction()

            # Grad. penalty: build interpolated image:
            epsilon = tf.random.uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
            x_interpolated = epsilon * self.unpaired_masks + (1 - epsilon) * synthetic_fake_replay
            self.real_fake_replay_interpolated = x_interpolated[..., 1:]
            model_interp = discriminator.build(self.real_fake_replay_interpolated, reuse=True)
            self.disc_pred_replay_interpolated = model_interp.get_prediction()

        # --------------
        # Combined model used for TTT
        with tf.variable_scope('Adaptor'):
            adaptor_test = adaptor.build(incoming=self.sup_input_data, reuse=True)
            adapted_test_input = adaptor_test.get_prediction()
        with tf.variable_scope('Segmentor'):
            unet_test = unet.build(adapted_test_input, reuse=True)
            pred_soft = unet_test.get_prediction(softmax=True)
            # code = unet_test.get_code()
        with tf.variable_scope('Discriminator'):
            disc_input = pred_soft[..., 1:]
            model_test_fake = discriminator.build(disc_input, reuse=True, zero_noise=True)
            self.test_pred_fake = model_test_fake.get_prediction()

        self.input = self.sup_input_data
        self.adapted_input = adapted_sup_input
        self.delta_input_sup = tf.abs(self.adapted_input - self.input)
        self.soft_prediction = self.sup_pred_mask_soft
        self.prediction = self.sup_pred_mask_oh
        self.ground_truth = self.sup_output_mask

    def define_losses(self):
        """
        Define loss function for each task.
        """
        # _______
        # Weighted Cross Entropy loss:
        with tf.variable_scope('WXEntropy_loss'):
            self.xentropy_loss = weighted_cross_entropy(y_pred=self.sup_pred_mask_soft,
                                                        y_true=self.sup_output_mask,
                                                        num_classes=self.n_classes)
        self.sup_loss = self.xentropy_loss

        # _______
        # Mask Discriminator loss:
        with tf.variable_scope('Discriminator_loss'):
            self.disc_loss_real = self.gan.discriminator_real_loss(self.disc_pred_real)
            self.disc_loss_fake = self.gan.discriminator_fake_loss(self.disc_pred_fake)
            self.adv_gen_loss = self.gan.generator_loss(self.disc_pred_fake)
            self.adv_disc_loss = self.disc_loss_real + self.disc_loss_fake

            self.disc_loss_synth_fake = self.gan.discriminator_fake_loss(self.disc_pred_synth_fake)
            if self.args.use_fake_anchors:
                self.adv_disc_loss += self.disc_loss_synth_fake

            self.gradient_penalty = gan_losses.gradient_penalty(
                x_interpolated=self.real_fake_interpolated,
                disc_pred_interpolated=self.disc_pred_interpolated)

            self.adv_disc_loss += self.gradient_penalty

        with tf.variable_scope('Discriminator_replay_loss'):
            replay_disc_loss = self.gan.discriminator_fake_loss(self.disc_pred_replay)

            self.gradient_penalty_replay = gan_losses.gradient_penalty(
                x_interpolated=self.real_fake_replay_interpolated,
                disc_pred_interpolated=self.disc_pred_replay_interpolated)

            replay_disc_loss += self.gradient_penalty_replay

        # define losses for supervised, unsupervised and frame prediction steps:
        w_adv = 0.1
        eps = 1e-16
        # find dynamic weight to balance current loss:
        generator_loss = self.adv_gen_loss
        w_dynamic = tf.abs(tf.stop_gradient(self.sup_loss / (generator_loss + eps)))

        self.generator_loss = w_adv * w_dynamic * self.adv_gen_loss
        self.discriminator_loss = self.adv_disc_loss
        self.discriminator_loss_replay = replay_disc_loss

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
            # Adversarial loss contributes:
            with tf.name_scope("DiscriminatorOptimizer"):
                disc_vars = [var for var in tf.trainable_variables() if var.name.startswith("Discriminator")]
                disc_optimizer = tf.train.AdamOptimizer(self.lr)
                disc_grads_and_vars = disc_optimizer.compute_gradients(self.discriminator_loss, var_list=disc_vars)
                train_op_discriminator = disc_optimizer.apply_gradients(disc_grads_and_vars,
                                                                        global_step=self.g_train_step)

                # Experience Replay:
                disc_grads_and_vars = disc_optimizer.compute_gradients(self.discriminator_loss_replay, var_list=disc_vars)
                train_op_replay = disc_optimizer.apply_gradients(disc_grads_and_vars, global_step=self.g_train_step)

            # In classical GAN architectures, you update discriminator and generator sequentially not in parallel.
            # Since I'm going to use tf.group() I will control dependencies before the updates
            with tf.name_scope("GeneratorOptimizer"):
                with tf.control_dependencies([train_op_discriminator]):  # , self.train_op_sup
                    gen_vars = [var for var in tf.trainable_variables()
                                if (var.name.startswith("Adaptor") or var.name.startswith("Segmentor"))]
                    gen_optimizer = tf.train.AdamOptimizer(self.lr)
                    gen_grads_and_vars = gen_optimizer.compute_gradients(self.generator_loss, var_list=gen_vars)
                    train_op_generator = gen_optimizer.apply_gradients(gen_grads_and_vars, global_step=self.g_train_step)

            # -----------------
            # Supervised segmentation loss
            with tf.name_scope("GeneratorOptimizer"):
                train_op_sup = gen_optimizer.minimize(self.sup_loss)

        # train op generator has dependencies on the rest
        # self.train_op = train_op_generator
        self.train_op = tf.group(train_op_sup, train_op_generator)
        self.train_op_replay = tf.group(train_op_sup, train_op_generator, train_op_replay)

        self.optimizer = gen_optimizer

    def define_eval_metrics(self):
        """
        Evaluate the model on the current batch
        """
        with tf.variable_scope('Dice_sup'):
            self.dice_sup = dice_coe(output=self.sup_pred_mask_oh[..., 1:],
                                     target=self.sup_output_mask[..., 1:])

    def define_summaries(self):
        """
        Create summaries to write on TensorBoard
        """
        # Scalar summaries:
        with tf.name_scope('Supervised_loss'):
            tr_dice_loss = tf.summary.scalar('train/sup_loss', self.sup_loss)
            val_dice_loss = tf.summary.scalar('validation/sup_loss', self.sup_loss)

        with tf.name_scope('Adversarial_loss'):
            adv_train_list = list()
            adv_train_list.append(tf.summary.scalar('train/adv_gen_loss', self.adv_gen_loss))
            adv_train_list.append(tf.summary.scalar('train/disc_loss_real', self.disc_loss_real))
            adv_train_list.append(tf.summary.scalar('train/disc_loss_fake', self.disc_loss_fake))
            if self.args.use_fake_anchors:
                adv_train_list.append(tf.summary.scalar('train/disc_loss_synth_fake', self.disc_loss_synth_fake))

            adv_valid_list = list()
            adv_valid_list.append(tf.summary.scalar('valid/adv_gen_loss', self.adv_gen_loss))
            adv_valid_list.append(tf.summary.scalar('valid/disc_loss_real', self.disc_loss_real))
            adv_valid_list.append(tf.summary.scalar('valid/disc_loss_fake', self.disc_loss_fake))
            if self.args.use_fake_anchors:
                adv_valid_list.append(tf.summary.scalar('valid/disc_loss_synth_fake', self.disc_loss_synth_fake))

        # Image summaries:
        with tf.name_scope('0_Input'):
            img_inp_s = tf.summary.image('0_input_sup', self.sup_input_data, max_outputs=3)
            img_ad_inp_s = tf.summary.image('1_adapted_input_sup', self.adapted_input, max_outputs=3)
            img_delta_inp_s = tf.summary.image('2_delta_input_sup', self.delta_input_sup, max_outputs=3)
        with tf.name_scope('1_Pred_Segmentation'):
            img_pred_mask = tf.summary.image('0_pred_mask', from_one_hot_to_rgb(self.sup_pred_mask_oh), max_outputs=3)
        with tf.name_scope('2_True_Segmentation'):
            img_mask = tf.summary.image('0_gt_mask', from_one_hot_to_rgb(self.sup_output_mask), max_outputs=3)
        with tf.name_scope('9_TEST_RESULTS'):
            img_test_results = list()
            img_test_results.append(tf.summary.image('0_input_sup', self.sup_input_data, max_outputs=3))
            img_test_results.append(tf.summary.image('1_pred_segm', from_one_hot_to_rgb(self.sup_pred_mask_oh), max_outputs=3))

        # merging all scalar summaries:
        sup_train_scalar_summaries = [tr_dice_loss]
        sup_train_scalar_summaries.extend(adv_train_list)
        sup_valid_scalar_summaries = [val_dice_loss]
        sup_valid_scalar_summaries.extend(adv_valid_list)
        sup_test_images_summaries = img_test_results

        self.sup_train_scalar_summary_op = tf.summary.merge(sup_train_scalar_summaries)
        self.sup_valid_scalar_summary_op = tf.summary.merge(sup_valid_scalar_summaries)

        all_train_summaries = []
        all_train_summaries.extend(sup_train_scalar_summaries)
        self.all_train_scalar_summary_op = tf.summary.merge(all_train_summaries)

        # _______________________________
        # merging all images summaries:
        sup_valid_images_summaries = [img_inp_s, img_ad_inp_s, img_delta_inp_s,
                                      img_mask, img_pred_mask]

        self.sup_valid_images_summary_op = tf.summary.merge(sup_valid_images_summaries)
        self.sup_test_images_summary_op = tf.summary.merge(sup_test_images_summaries)

        # ---- #
        if self.tensorboard_verbose:
            _vars = [v.name for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'kernel' in v.name]
            weights_summary = [tf.summary.histogram(v, tf.get_default_graph().get_tensor_by_name(v)) for v in _vars]
            self.weights_summary = tf.summary.merge(weights_summary)

    def _train_all_op(self, sess, writer, step):
        _, sl, dl_sfake, dl_real, dl_fake, gl, scalar_summaries \
            = sess.run([self.train_op,
                        self.sup_loss,
                        self.disc_loss_synth_fake,
                        self.disc_loss_real,
                        self.disc_loss_fake,
                        self.adv_gen_loss,
                        self.all_train_scalar_summary_op],
                       feed_dict={self.is_training: True})

        if random.randint(0, self.train_summaries_skip) == 0:
            writer.add_summary(scalar_summaries, global_step=step)

        return sl, dl_real, dl_fake, dl_sfake, gl

    def train_one_epoch(self, sess, iterator_init_list, writer, step, caller, seed):
        """ train the model for one epoch. """
        start_time = time.time()

        # setup progress bar
        self.progress_bar.attach()
        self.progress_bar.monitor_progress()

        # initialize data set iterators:
        for init in iterator_init_list:
            sess.run(init, feed_dict={self.global_seed: seed})

        # iterators:
        n_batches = 0

        # total losses:
        metrics = {
            'supervised_loss': 0,
            'unsupervised_loss': 0,
            'discriminator_loss_real': 0,
            'discriminator_loss_syth_fake': 0,
            'discriminator_loss_fake': 0,
            'generator_loss': 0
        }

        try:
            while True:
                self.progress_bar.monitor_progress()

                caller.on_batch_begin(training_state=True, **self.callbacks_kwargs)

                sl, dl_real, dl_fake, dl_sfake, gl = self._train_all_op(sess, writer, step)
                metrics['supervised_loss'] += sl
                metrics['discriminator_loss_real'] += dl_real
                metrics['discriminator_loss_syth_fake'] += dl_sfake
                metrics['discriminator_loss_fake'] += dl_fake
                metrics['generator_loss'] += gl
                metrics['unsupervised_loss'] += (dl_sfake + dl_real + dl_fake + gl)
                step += 1

                n_batches += 1
                if (n_batches % self.skip_step) == 0 and self.verbose:
                    print('\r  ...training over batch {1}: {0} batch_sup_loss = {2:.4f}\tbatch_unsup_loss = {3:.4f} {0}'
                          .format(' ' * 3, n_batches, metrics['supervised_loss'], metrics['unsupervised_loss']), end='\n')

                caller.on_batch_end(training_state=True, **self.callbacks_kwargs)

        except tf.errors.OutOfRangeError:
            # End of the epoch. Compute statistics here:
            delta_t = time.time() - start_time

            for key in ['discriminator_loss_real', 'discriminator_loss_fake', 'discriminator_loss_syth_fake', 'generator_loss']:
                avg_loss = metrics[key] / n_batches
                value = summary_pb2.Summary.Value(tag="AdversarialTrain/{0}".format(key), simple_value=avg_loss)
                summary = summary_pb2.Summary(value=[value])
                writer.add_summary(summary, global_step=step)

        # update global epoch counter:
        sess.run(self.increase_g_epoch)
        sess.run(self.update_g_train_step, feed_dict={'update_value:0': step})

        # detach progress bar and update last time of arrival:
        self.progress_bar.detach()
        self.progress_bar.update_lta(delta_t)

        print('\033[31m  TRAIN\033[0m:{0}{0} average loss = {1:.4f} {0} Took: {2:.3f} seconds'
              .format(' ' * 3, avg_loss, delta_t))
        return step

    def _eval_all_op(self, sess, writer, step):
        sl, dice_score, dl_sfake, dl_real, dl_fake, gl, sup_sc_summ, sup_im_summ = \
            sess.run([self.sup_loss,
                      self.dice_sup,
                      self.disc_loss_synth_fake,
                      self.disc_loss_real,
                      self.disc_loss_fake,
                      self.adv_gen_loss,
                      self.sup_valid_scalar_summary_op,
                      self.sup_valid_images_summary_op],
                     feed_dict={self.is_training: False})
        writer.add_summary(sup_sc_summ, global_step=step)
        writer.add_summary(sup_im_summ, global_step=step)

        return sl, dice_score, dl_real, dl_fake, dl_sfake, gl

    def eval_once(self, sess, iterator_init_list, writer, step, caller):
        """ Eval the model once """
        start_time = time.time()

        # initialize data set iterators:
        for init in iterator_init_list:
            sess.run(init)

        # iterators:
        n_batches = 0

        # total losses:
        metrics = {
            'supervised_loss': 0,
            'dice_score': 0,
            'unsupervised_loss': 0,
            'discriminator_loss_real': 0,
            'discriminator_loss_syth_fake': 0,
            'discriminator_loss_fake': 0,
            'generator_loss': 0
        }
        try:
            while True:
                caller.on_batch_begin(training_state=False, **self.callbacks_kwargs)

                sl, dice_score, dl_real, dl_fake, dl_sfake, gl = self._eval_all_op(sess, writer, step)
                metrics['dice_score'] += dice_score
                metrics['supervised_loss'] += sl
                metrics['discriminator_loss_real'] += dl_real
                metrics['discriminator_loss_syth_fake'] += dl_sfake
                metrics['discriminator_loss_fake'] += dl_fake
                metrics['generator_loss'] += gl
                metrics['unsupervised_loss'] += (dl_sfake + dl_real + dl_fake + gl)
                step += 1

                n_batches += 1
                caller.on_batch_end(training_state=False, **self.callbacks_kwargs)

        except tf.errors.OutOfRangeError:
            # End of the validation set. Compute statistics here:
            avg_dice = metrics['dice_score'] / n_batches
            dice_loss = 1.0 - avg_dice
            delta_t = time.time() - start_time

            value = summary_pb2.Summary.Value(tag="Dice/validation/dice_no_bgd_avg", simple_value=avg_dice)
            summary = summary_pb2.Summary(value=[value])
            writer.add_summary(summary, global_step=step)

            for key in ['discriminator_loss_real', 'discriminator_loss_fake', 'discriminator_loss_syth_fake', 'generator_loss']:
                avg_loss = metrics[key] / n_batches
                value = summary_pb2.Summary.Value(tag="AdversarialValidation/{0}".format(key), simple_value=avg_loss)
                summary = summary_pb2.Summary(value=[value])
                writer.add_summary(summary, global_step=step)

        # update global epoch counter:
        sess.run(self.update_g_valid_step, feed_dict={'update_value:0': step})

        print('\033[31m  VALIDATION\033[0m:  average loss = {1:.4f} {0} Took: {2:.3f} seconds'
              .format(' ' * 3, avg_loss, delta_t))
        return step, dice_loss

    def test_once(self, sess, sup_test_init, writer, step, caller):
        """ Test the model once """
        start_time = time.time()

        # initialize data set iterators:
        sess.run(sup_test_init)

        total_dice_score = 0
        n_batches = 0
        try:
            while True:
                caller.on_batch_begin(training_state=False, **self.callbacks_kwargs)

                dice_sup, images_summaries = sess.run([self.dice_sup, self.sup_test_images_summary_op],
                                                      feed_dict={self.is_training: False})
                total_dice_score += dice_sup
                writer.add_summary(images_summaries, global_step=step)

                n_batches += 1

        except tf.errors.OutOfRangeError:
            # End of the test set. Compute statistics here:
            avg_dice = total_dice_score / n_batches
            delta_t = time.time() - start_time

            step += 1
            value = summary_pb2.Summary.Value(tag="ZZZ_TEST/test/dice_no_bgd_avg", simple_value=avg_dice)
            summary = summary_pb2.Summary(value=[value])
            writer.add_summary(summary, global_step=step)
            pass

        # update global epoch counter:
        sess.run(self.update_g_test_step, feed_dict={'update_value:0': step})

        print('\033[31m  TEST\033[0m:{0}{0} \033[1;33m average dice = {1:.4f}\033[0m on \033[1;33m{2}\033[0m batches '
              '{0} Took: {3:.3f} seconds'.format(' ' * 3, avg_dice, n_batches, delta_t))
        return step

    def train(self, n_epochs):
        """ The train function alternates between training one epoch and evaluating """
        if self.verbose:
            print("\nStarting network training... Number of epochs to train: \033[94m{0}\033[0m".format(n_epochs))
            print("Tensorboard verbose mode: \033[94m{0}\033[0m".format(self.tensorboard_verbose))
            print("Tensorboard dir: \033[94m{0}\033[0m".format(self.graph_dir))
            print("Data augmentation: \033[94m{0}\033[0m, Data standardization: \033[94m{1}\033[0m."
                  .format(self.augment, self.standardize))

        utils.safe_mkdir(self.checkpoint_dir)
        utils.safe_mkdir(self.history_log_dir)
        writer = tf.summary.FileWriter(self.graph_dir, tf.get_default_graph())

        # config for the session: allow growth for GPU to avoid OOM when other processes are running
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            saver = tf.train.Saver(max_to_keep=2)  # keep_checkpoint_every_n_hours=2
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.last_checkpoint_dir + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            trained_epochs = self.g_epoch.eval()

            if self.verbose:
                print("Model already trained for \033[94m{0}\033[0m epochs.".format(trained_epochs))
            t_step = self.g_train_step.eval()  # global step for train
            v_step = self.g_valid_step.eval()  # global step for validation
            test_step = self.g_test_step.eval()  # global step for test

            # Define a caller to call the callbacks
            self.callbacks_kwargs.update({'sess': sess, 'cnn': self})
            caller = tf_callbacks.ChainCallback(callbacks=self.callbacks)
            caller.on_train_begin(training_state=True, **self.callbacks_kwargs)

            # trick to find performance bugs: this will raise an exception if any new node is inadvertently added to the
            # graph. This will ensure that I don't add many times the same node to the graph (which could be expensive):
            tf.get_default_graph().finalize()

            # saving callback:
            self.callbacks_kwargs['es_loss'] = 100  # some random initialization

            for epoch in range(n_epochs):
                ep_str = str(epoch + 1) if (trained_epochs == 0) else '({0}+) '.format(trained_epochs) + str(epoch + 1)
                print('_' * 40 + '\n\033[1;33mEPOCH {0}\033[0m - {1} : '.format(ep_str, self.run_id))
                caller.on_epoch_begin(training_state=True, **self.callbacks_kwargs)

                global_ep = sess.run(self.g_epoch)
                self.callbacks_kwargs['es_loss'] = sess.run(self.last_val_loss)
                # sess.run(self.update_lr)

                seed = global_ep

                # TRAIN MODE ------------------------------------------
                iterator_init_list = [self.sup_train_init,
                                      self.disc_train_init,
                                      self.unsup_train_init]
                t_step = self.train_one_epoch(sess, iterator_init_list, writer, t_step, caller, seed)

                # VALIDATION MODE ------------------------------------------
                # if global_ep >= 400 or not ((global_ep + 1) % 15):  # when to evaluate the model
                ep_offset = self.args.validation_offset
                if global_ep >= ep_offset or not ((global_ep + 1) % 10):  # when to evaluate the model
                    iterator_init_list = [self.sup_valid_init,
                                          self.disc_valid_init,
                                          self.unsup_valid_init]
                    v_step, val_loss = self.eval_once(sess, iterator_init_list, writer, v_step, caller)

                    self.callbacks_kwargs['es_loss'] = val_loss
                    sess.run(self.update_last_val_loss, feed_dict={'best_val_loss_value:0': val_loss})

                # ----------------------------------------------------
                # # save updated variables and weights
                # saver.save(sess, self.checkpoint_dir + '/checkpoint', t_step)

                if self.tensorboard_verbose and (global_ep % 10 == 0) or (global_ep == 0):
                    # writing summary for the weights:
                    summary = sess.run(self.weights_summary)
                    writer.add_summary(summary, global_step=t_step)

                try:
                    caller.on_epoch_end(training_state=True, **self.callbacks_kwargs)
                except EarlyStoppingException:
                    utils.print_yellow_text('\nEarly stopping...\n')
                    break
                except NeedForTestException:
                    # early stopping criterion: save the model
                    saver.save(sess, self.checkpoint_dir + '/checkpoint', t_step)

            caller.on_train_end(training_state=True, **self.callbacks_kwargs)

            # end of the training: save the current weights in a new sub-directory
            utils.safe_mkdir(self.checkpoint_dir + '/last_model')
            saver.save(sess, self.checkpoint_dir + '/last_model/checkpoint', t_step)

            # ----------------------------------------------------------
            # final testing phase

            # load best model and do a test:
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoint_dir + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            _ = self.test_once(sess, self.sup_test_init, writer, test_step, caller)

            # tf.get_default_graph()._unsafe_unfinalize()
            # self.test(sess)

        writer.close()
