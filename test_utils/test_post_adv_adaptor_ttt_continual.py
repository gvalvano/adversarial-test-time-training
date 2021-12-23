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

import tensorflow as tf
tf.random.set_random_seed(1234)
import numpy as np
np.random.seed(1234)
import random
random.seed(1234)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
from idas.utils import safe_mkdir
import PIL.Image
from idas.tf_utils import from_one_hot_to_rgb
from medpy.metric.binary import hd
import utils_sql as usql
import config
import datetime
import time
import cv2
from idas.utils import Colors


args = config.define_flags()


def npy_entropy_loss(v, eps=1e-16):
    c = v.shape[-1]
    k = -1 / np.log2(float(c))
    entropy = k * np.sum(v * np.log2(v + eps), axis=-1)
    mean_entropy = np.median(entropy, axis=(1, 2))
    return np.mean(mean_entropy)


def hausdorff_distance(mask1, mask2):
    """Compute the average Hausdorff distance for the patient (in pixels), between mask1 and mask2."""

    def _py_hd(m1, m2):
        """Python function to compute HD between the two n-dimensional masks"""
        m1, m2 = np.array(m1), np.array(m2)
        num_elems = len(m1)
        assert len(m2) == num_elems
        num_classes = m1.shape[-1]
        hd_list = []
        for c in range(num_classes):
            try:
                hd_list.append(hd(m1[..., c], m2[..., c]))
            except:
                pass
        if len(hd_list) == 0:
            return np.min([m1.shape[0], m1.shape[1]]) // 2
        return np.mean(hd_list)

    # map _py_hd(.) to every element on the batch axis:
    tf_hd = tf.py_function(func=_py_hd, inp=[mask1, mask2],
                           Tout=[tf.float32], name='hausdorff_distance')

    # return the average HD in the batch:
    return tf.reduce_mean(tf_hd)
    # return tf.reduce_max(tf_hd)


def dice_coe(output, target, axis=(1, 2), smooth=1e-12):
    """Compute the average Dice score between output and target segmentation masks."""
    intersection = tf.reduce_sum(output * target, axis=axis)
    a = tf.reduce_sum(output, axis=axis)
    b = tf.reduce_sum(target, axis=axis)
    score = (2. * intersection + smooth) / (a + b + smooth)
    score = tf.reduce_mean(score, name='dice_coe')
    return score


def iou_coe(output, target, axis=(1, 2), smooth=1e-12):
    """Compute the average IOU score between output and target segmentation masks."""
    intersection = tf.reduce_sum(output * target, axis=axis)

    a = tf.reduce_sum(output * output, axis=axis)
    b = tf.reduce_sum(target * target, axis=axis)

    union = a + b - intersection
    score = (intersection + smooth) / (union + smooth)
    score = tf.reduce_mean(score)
    return score


def get_channel(incoming, idx):
    return tf.expand_dims(incoming[..., idx], -1)


def data_augmentation(incoming):

    n_samples, rows, cols, channels = incoming.shape
    augmented = []
    for image in incoming:
        # open cv requires swapped rows and cols for the center:
        center = (cols // 2, rows // 2)

        # sample transformation parameters for the current element of the batch
        minval = - 0.1 * rows
        maxval = 0.1 * cols
        tx, ty = np.random.randint(low=minval, high=maxval, size=2)
        scale = np.random.uniform(low=0.98, high=1.02)
        angle = np.random.uniform(low=-90.0, high=90.0)
        # create transformation matrices:
        m1 = np.float32([[1, 0, tx], [0, 1, ty]])
        m2 = cv2.getRotationMatrix2D(center, angle, scale)

        # apply transformation matrix for each channel
        out = []
        for i in range(channels):
            curr_slice = image[..., i]
            transform_slice = cv2.warpAffine(curr_slice, m1, (cols, rows))
            transform_slice = cv2.warpAffine(transform_slice, m2, (cols, rows))
            out.append(np.expand_dims(transform_slice, axis=-1))

        augmented.append(np.concatenate(out, axis=-1))

    return np.array(augmented)


def _test_time_training(image, sess, model, test_training_op, adaptation_loss, n_steps, threshold, saver,
                        do_data_augmentation=False, adaptation_loss_before=1e10):
    n_ttt_step, ns, patience = 0, 0, 0
    update_steps = 0
    min_loss = adaptation_loss_before
    x = image

    # current predictions:
    image_adapt, y_pred, y_pred_soft, adaptation_loss_after = \
        sess.run([model.adapted_input, model.prediction, model.soft_prediction, adaptation_loss],
                 feed_dict={model.input: image, model.is_training: False})

    # test-time training:
    while (min_loss > threshold) and (n_ttt_step < n_steps):
        # test-time training:
        n_ttt_step += 1
        patience += 1
        if do_data_augmentation:
            x = data_augmentation(image)
        _ = sess.run(test_training_op, feed_dict={model.input: x, model.is_training: True})
        loss = sess.run(adaptation_loss, feed_dict={model.input: image, model.is_training: False})

        if loss < min_loss:
            patience = 0
            update_steps += 1
            min_loss = loss
            ns = n_ttt_step
            image_adapt, y_pred, y_pred_soft, adaptation_loss_after = \
                sess.run([model.adapted_input, model.prediction, model.soft_prediction, adaptation_loss],
                         feed_dict={model.input: image, model.is_training: False})

            # save new model
            for s in saver:
                s.save(sess, model.checkpoint_dir + '/checkpoint', ns)

        if patience == 200:
            print('     | Early stopping of TTT, iteration: {0}'.format(n_ttt_step))
            break

    return (image_adapt, y_pred, y_pred_soft, adaptation_loss_after), (min_loss, ns)


def color_delta(v1, v2, reverse=False):
    blue_text = lambda v: '{0}{1:.2f}{2}'.format(Colors.OKBLUE, v, Colors.ENDC)
    red_text = lambda v: '{0}{1:.2f}{2}'.format(Colors.FAIL, v, Colors.ENDC)
    gray_text = lambda v: '{0}{1:.2f}{2}'.format('\033[37m', v, Colors.ENDC)
    if reverse:
        bfr = blue_text(v1) if (v1 > v2) else red_text(v1) if (v1 < v2) else gray_text(v1)
        aft = blue_text(v2) if (v1 > v2) else red_text(v2) if (v1 < v2) else gray_text(v2)
    else:
        bfr = blue_text(v1) if (v1 < v2) else red_text(v1) if (v1 > v2) else gray_text(v1)
        aft = blue_text(v2) if (v1 < v2) else red_text(v2) if (v1 > v2) else gray_text(v2)
    delta = '{0:.3f}'.format(v2 - v1)
    return bfr, aft, delta


def test_model(sess, saver, ckpt, model, do_data_augmentation=False, n_steps=10, n_images=3):
    """ Test the model once """

    y_true_plch = tf.placeholder(tf.float32, [None, model.input_size[0], model.input_size[1], model.n_classes])
    y_pred_plch = tf.placeholder(tf.float32, [None, model.input_size[0], model.input_size[1], model.n_classes])
    y_pred_ttt_plch = tf.placeholder(tf.float32, [None, model.input_size[0], model.input_size[1], model.n_classes])
    y_true_rgb = from_one_hot_to_rgb(y_true_plch, background='white')
    y_pred_rgb = from_one_hot_to_rgb(y_pred_plch, background='white')
    y_pred_ttt_rgb = from_one_hot_to_rgb(y_pred_ttt_plch, background='white')

    # global and class-specific metrics:
    dice_test_list = [dice_coe(output=y_pred_plch[..., 1:], target=y_true_plch[..., 1:])]
    dice_test_list_cls = [dice_coe(output=get_channel(y_pred_plch, i), target=get_channel(y_true_plch, i)) for i in range(model.n_classes)]
    iou_test_list = [iou_coe(output=y_pred_plch[..., 1:], target=y_true_plch[..., 1:])]
    iou_test_list_cls = [iou_coe(output=get_channel(y_pred_plch, i), target=get_channel(y_true_plch, i)) for i in range(model.n_classes)]
    hd_test_list_cls = [hausdorff_distance(y_pred_plch[..., i], y_true_plch[..., i]) for i in range(model.n_classes)]

    # -------------------------------------------------------------------------------------------
    def _test_model(n_cls, n_img, img, pred, true):

        # global list of test to do:
        test_list = []
        test_list.extend(dice_test_list)
        test_list.extend(dice_test_list_cls)
        test_list.extend(iou_test_list)
        test_list.extend(iou_test_list_cls)
        test_list.extend(hd_test_list_cls)

        # assign results to each value
        if n_img > 0:
            # add also images to the test list
            image_list = [y_pred_rgb, y_true_rgb]
            test_list.extend(image_list)

        results = sess.run(test_list, feed_dict={y_pred_plch: pred, y_true_plch: true})
        idx = 0
        dc = results[idx]
        dc_per_class = []
        for c in range(n_cls):
            idx += 1
            dc_per_class.append(results[idx])

        idx += 1
        iu = results[idx]
        iu_per_class = []
        for c in range(n_cls):
            idx += 1
            iu_per_class.append(results[idx])

        # this time start from 0 as we don't have hd inside results
        hd_per_class = []
        for c in range(n_cls):
            idx += 1
            hd_per_class.append(results[idx])
        hd = np.mean(hd_per_class)
        if n_img > 0:
            idx += 1
            imgs_tuple = [img]
            imgs_tuple.extend(results[idx:])
        else:
            imgs_tuple = None
        return dc, dc_per_class, iu, iu_per_class, hd, hd_per_class, imgs_tuple

    # -------------------------------------------------------------------------------------------

    # Test
    n_classes = model.n_classes
    sess.run(model.test_init)  # initialize data set iterator on test set:

    # initialize a dictionary with the metrics
    metrics = dict()
    metrics['adaptation_loss_before'] = list()
    metrics['adaptation_loss_after'] = list()
    metrics['entropy_before'] = list()
    metrics['entropy_after'] = list()
    metrics['dice_before'] = list()
    metrics['dice_after'] = list()
    metrics['iou_before'] = list()
    metrics['iou_after'] = list()
    metrics['hd_before'] = list()
    metrics['hd_after'] = list()

    adaptation_loss = model.adaptation_loss
    adaptation_threshold = model.adaptation_threshold
    test_training_op = model.test_time_train_op

    n_batches = 0
    img_list = []
    ns_list = []
    try:
        while True:
            # print('Reloading model from checkpoint...')
            for s in saver:
                s.restore(sess, ckpt.model_checkpoint_path)

            if ((n_batches + 1) % 5 == 0) or (n_batches == 0):
                ts = time.time()
                s = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                print('New test image (N = {0}) - {1}'.format(n_batches + 1, s))
            image, label = sess.run([model.input_images, model.ground_truth])
            batch_max = 30 # TODO 32
            if len(image) > batch_max:
                # prevents OOM on sample #40 :
                image = image[:batch_max, ...]
                label = label[:batch_max, ...]

            # --------------------------------------------------------
            # default test:
            image_adapt, y_pred, y_pred_soft, adaptation_loss_before = \
                sess.run([model.adapted_input, model.prediction, model.soft_prediction, adaptation_loss],
                         feed_dict={model.input: image, model.is_training: False})

            dice_before, dice_per_class, iou_before, iou_per_class, hdist_before, hdist_per_class, imgs = \
                _test_model(n_classes, n_images, image, y_pred, label)
            entropy_before = npy_entropy_loss(y_pred_soft)

            #  --------------------------------------------------------
            if adaptation_loss_before > adaptation_threshold:
                print('  > Perform TTT on N = {0}.'.format(n_batches + 1))
                (image_adapt_ttt, y_pred_ttt, y_pred_ttt_soft, adaptation_loss_after), (adaptation_loss_after, ns) = \
                    _test_time_training(image, sess, model, test_training_op, adaptation_loss, n_steps,
                                        adaptation_threshold, saver, do_data_augmentation, adaptation_loss_before)
                print('     | Starting loss = {0:.4f}; threshold = {1:.4f}; final loss = {2:.4f} (num. TTT steps = {3})'
                      .format(adaptation_loss_before, adaptation_threshold, adaptation_loss_after, ns))
                ns_list.append(ns)

                dice_after, dice_per_class, iou_after, iou_per_class, hdist_after, hdist_per_class, imgs = \
                    _test_model(n_classes, n_images, image, y_pred_ttt, label)

                bfr, aft, delta = color_delta(100 * dice_before, 100 * dice_after)
                print('     | Dice before: {0}, Dice after: {1}   (delta = {2})'.format(bfr, aft, delta))

                entropy_after = npy_entropy_loss(y_pred_ttt_soft)
                # bfr, aft, delta = color_delta(1e7 * entropy_before, 1e7 * entropy_after, reverse=True)
                # print('     | Entropy before: 1e-7 * {0}, Entropy after: 1e-7 * {1}   (delta = 1e-7 * {2})'.format(bfr, aft, delta))
            else:
                ns_list.append(0)
                dice_after, iou_after, hdist_after = dice_before, iou_before, hdist_before
                adaptation_loss_after = adaptation_loss_before
                entropy_after = entropy_before
                y_pred_ttt = y_pred
                image_adapt_ttt = image_adapt

            label, y_pred, y_pred_ttt = sess.run([y_true_rgb, y_pred_rgb, y_pred_ttt_rgb],
                                                 feed_dict={y_true_plch: label,
                                                            y_pred_plch: y_pred,
                                                            y_pred_ttt_plch: y_pred_ttt})

            # add images to the list:
            image_diff = image_adapt_ttt - image
            if imgs is not None: img_list.append([image, y_pred, image_adapt, image_diff, y_pred_ttt, label])

            #  --------------------------------------------------------
            # go on with statistics:
            n_batches += 1
            n_images -= 1

            # save results
            metrics['adaptation_loss_before'].append(adaptation_loss_before)
            metrics['adaptation_loss_after'].append(adaptation_loss_after)
            metrics['entropy_before'].append(entropy_before)
            metrics['entropy_after'].append(entropy_after)
            metrics['dice_before'].append(dice_before)
            metrics['dice_after'].append(dice_after)
            metrics['iou_before'].append(iou_before)
            metrics['iou_after'].append(iou_after)
            metrics['hd_before'].append(hdist_before)  # hd = hausdorff_distance
            metrics['hd_after'].append(hdist_after)  # hd = hausdorff_distance

    except tf.errors.OutOfRangeError:
        avg_dice_before, std_dice_before = np.mean(metrics['dice_before']), np.std(metrics['dice_before'])
        avg_dice_after, std_dice_after = np.mean(metrics['dice_after']), np.std(metrics['dice_after'])
        avg_iou_before, std_iou_before = np.mean(metrics['iou_before']), np.std(metrics['iou_before'])
        avg_iou_after, std_iou_after = np.mean(metrics['iou_after']), np.std(metrics['iou_after'])
        avg_hd_before, std_hd_before = np.mean(metrics['hd_before']), np.std(metrics['hd_before'])
        avg_hd_after, std_hd_after = np.mean(metrics['hd_after']), np.std(metrics['hd_after'])

    return \
        metrics['adaptation_loss_before'], metrics['adaptation_loss_after'], \
        metrics['entropy_before'], metrics['entropy_after'], \
        avg_dice_before, std_dice_before, metrics['dice_before'], \
        avg_dice_after, std_dice_after, metrics['dice_after'], \
        avg_iou_before, std_iou_before, metrics['iou_before'], \
        avg_iou_after, std_iou_after, metrics['iou_after'], \
        avg_hd_before, std_hd_before, metrics['hd_before'], \
        avg_hd_after, std_hd_after, metrics['hd_after'], ns_list, img_list


def plot_batch(img_list, path_prefix):
    """Save batch of images tiled."""

    def _postprocess_image(img):
        """ from float range in about [-1, 1] to uint8 in [0, 255] """
        # rescale:
        img = img + abs(img.min())
        img = img / img.max()
        img = np.clip(255 * img, 0, 255)
        img = img.astype(np.uint8)
        return img

    def _safe_rgb(img):
        """ Converts grayscale image to rgb, if needed """
        if img.shape[-1] == 1:
            img = np.stack((np.squeeze(img, axis=-1),) * 3, axis=-1)
        return img

    def _tile(img_lst, n_rows):
        """Tile images for display."""
        x, yp, xa, xd, ypttt, yt = img_lst
        n_cols = 6  # one for each: x, yp, xa, xd, ypttt, yt
        assert x.shape == yp.shape
        assert x.shape == yt.shape
        h, w = x.shape[1], x.shape[2]

        # initialize and then fill empty array with input images:
        tiled = np.zeros((n_rows * h, n_cols * w, 3), dtype=x.dtype)
        for row_i in range(n_rows):
            x_i, yp_i, xa_i, xd_i, ypttt_i, yt_i = x[row_i], yp[row_i], xa[row_i], xd[row_i], ypttt[row_i], yt[row_i]
            tiled[row_i * h: (row_i + 1) * h, 0: w, :] = x_i
            tiled[row_i * h: (row_i + 1) * h, w: 2 * w, :] = yp_i
            tiled[row_i * h: (row_i + 1) * h, 2 * w: 3 * w, :] = xa_i
            tiled[row_i * h: (row_i + 1) * h, 3 * w: 4 * w, :] = xd_i
            tiled[row_i * h: (row_i + 1) * h, 4 * w: 5 * w, :] = ypttt_i
            tiled[row_i * h: (row_i + 1) * h, 5 * w:, :] = yt_i
        return tiled

    for i in range(len(img_list)):
        x_in, y_pred, x_adapt, x_diff, y_pred_ttt, y_true = img_list[i]
        x_in = _postprocess_image(x_in)
        x_in = _safe_rgb(x_in)
        x_adapt = _postprocess_image(x_adapt)
        x_adapt = _safe_rgb(x_adapt)
        x_diff = _postprocess_image(x_diff)
        x_diff = _safe_rgb(x_diff)

        rows = len(x_in)
        canvas = _tile([x_in, y_pred, x_adapt, x_diff, y_pred_ttt, y_true], rows)
        canvas = np.squeeze(canvas)
        PIL.Image.fromarray(canvas).save(os.path.join(path_prefix, 'test_batch{0}.png'.format(i)))


# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================


def test(model, sess, saver, ckpt, n_test_steps=1000, test_augmentation=False, n_images=2):
    print('-----------\nPerforming final test, before adding elements to SQL database...')
    test_adaptation_loss = 'discriminator_loss'
    epoch = sess.run(model.g_epoch)

    suffix = '_{0}_{1}'.format(model.args.n_sup_vols, model.args.split_number)
    run_id = model.run_id.rsplit(suffix)[0]
    r_id = run_id + '_Continual' + suffix

    print("run-id = \033[94m{0}\033[0m, database = \033[94m{1}\033[0m, table = \033[94m{2}\033[0m"
          .format(r_id, args.sql_db_name, args.table_name))
    print("Model trained for \033[94m{0}\033[0m epochs".format(epoch))
    print("TTT with maximum 'n_steps' = \033[94m{0}\033[0m".format(n_test_steps))
    print("Data augmentation = \033[94m{0}\033[0m".format(test_augmentation))
    print("-----------\n")

    # do a test:
    adaptation_loss_before, adaptation_loss_after, entropy_before, entropy_after, \
        avg_dice_before, std_dice_before, dice_list_before, \
        avg_dice_after, std_dice_after, dice_list_after, \
        avg_iou_before, std_iou_before, iou_list_before, \
        avg_iou_after, std_iou_after, iou_list_after, \
        avg_hd_before, std_hd_before, hd_list_before, \
        avg_hd_after, std_hd_after, hd_list_after, \
        ns_list, img_list = test_model(sess, saver, ckpt, model, n_steps=n_test_steps, n_images=n_images,
                                       do_data_augmentation=test_augmentation)

    # save the images:
    results_dir = args.results_dir
    modality = args.modality
    if modality == '':
        modality = None
    dataset_name = args.dataset_name if modality is None else '{0}_{1}'.format(args.dataset_name, modality)
    if modality is not None: assert modality in ['T1', 'T2', 'CT']
    safe_mkdir('{0}/results/{1}/{2}/images/'.format(results_dir, args.experiment_type, dataset_name))
    safe_mkdir('{0}/results/{1}/{2}/images/{3}'.format(results_dir, args.experiment_type, dataset_name, args.n_sup_vols))
    safe_mkdir('{0}/results/{1}/{2}/images/{3}/{4}'.format(results_dir, args.experiment_type, dataset_name, args.n_sup_vols, r_id))
    dest_dir = '{0}/results/{1}/{2}/images/{3}/{4}'.format(results_dir, args.experiment_type, dataset_name, args.n_sup_vols, r_id)
    plot_batch(img_list, path_prefix=dest_dir)

    with open(os.path.join(dest_dir, 'report.txt'), 'w') as f:
        report = 'Dataset: {0} \n Run-ID: {1}\n TTT steps: {2}\n'.format(model.args.dataset_name, run_id, n_test_steps)
        report += '\n Rows = different samples'
        report += '\n Columns = Input image, Prediction, Adapted Image, Adapted - Input image, Prediction after TTT, Ground Truth'
        f.write(report)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    print("\n" + "- " * 20)
    print(f"Results for run-id: {r_id}")
    print(f"\t | Average (standard dev.) Dice score:          "
          f"before={avg_dice_before} ({std_dice_before}),  "
          f"after={avg_dice_after} ({std_dice_after})")
    print(f"\t | Average (standard dev.) IoU score:           "
          f"before={avg_iou_before} ({std_iou_before}),  "
          f"after={avg_iou_after} ({std_iou_after})")
    print(f"\t | Average (standard dev.) Hausdorff Distance:  "
          f"before={avg_hd_before} ({std_hd_before}),  "
          f"after={avg_hd_after} ({std_hd_after})")
    print("- " * 20 + "\n")

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # insert values in results SQL database:

    # create a database connection
    database = args.sql_db_name
    table_name = args.table_name
    conn = usql.create_connection(database)

    print(f"Storing full test details and metrics in the database: {database}, table: {table_name}.")

    # create tables
    if conn is not None:
        print('\nCreating table...')
        statement = """ 
        CREATE TABLE IF NOT EXISTS {0} (
        id integer PRIMARY KEY,
        RUN_ID text NOT NULL,
        PERC text NOT NULL,
        SPLIT text NOT NULL,
        CONFIG text NOT NULL,
        EXPERIMENT_TYPE text NOT NULL,
        DATASET_NAME text NOT NULL,
        INPUT_SIZE text NOT NULL,
        EPOCH integer NOT NULL,
        TEST_ADAPTATION_LOSS text NOT NULL,
        TEST_ADAPTATION_THRESHOLD float NOT NULL,
        TEST_AUGMENTATION text NOT NULL,
        N_TEST_STEPS_MAX text NOT NULL,
        N_TEST_STEPS text NOT NULL,
        ADAPTATION_LOSS_LIST_BEFORE text NOT NULL,
        ADAPTATION_LOSS_LIST_AFTER text NOT NULL,
        ENTROPY_LIST_BEFORE text NOT NULL,
        ENTROPY_LIST_AFTER text NOT NULL,
        AVG_DICE_BEFORE float NOT NULL,
        STD_DICE_BEFORE float NOT NULL,
        DICE_LIST_BEFORE text NOT NULL,
        AVG_DICE_AFTER float NOT NULL,
        STD_DICE_AFTER float NOT NULL,
        DICE_LIST_AFTER text NOT NULL,
        AVG_IOU_BEFORE float NOT NULL,
        STD_IOU_BEFORE float NOT NULL,
        IOU_LIST_BEFORE text NOT NULL,
        AVG_IOU_AFTER float NOT NULL,
        STD_IOU_AFTER float NOT NULL,
        IOU_LIST_AFTER text NOT NULL,
        AVG_HD_BEFORE float NOT NULL,
        STD_HD_BEFORE float NOT NULL,
        HD_LIST_BEFORE text NOT NULL,
        AVG_HD_AFTER float NOT NULL,
        STD_HD_AFTER float NOT NULL,
        HD_LIST_AFTER text NOT NULL,
        TIMESTAMP text
        ); """.format(table_name)
        usql.create_table(conn, statement)
    else:
        print("\n\033[31m  Error! cannot create the database connection.\033[0m")
        raise

    with conn:
        print('Creating connection...')
        # get values and insert into table:
        values = [r_id, args.n_sup_vols, args.split_number, str(args),
                  args.experiment_type, args.dataset_name,
                  str(model.input_size),
                  int(epoch),
                  str(test_adaptation_loss),
                  str(model.adaptation_threshold),
                  str(test_augmentation),
                  str(n_test_steps),
                  str(ns_list),
                  str(adaptation_loss_before),
                  str(adaptation_loss_after),
                  str(entropy_before),
                  str(entropy_after),
                  float(avg_dice_before),
                  float(std_dice_before),
                  str(dice_list_before),
                  float(avg_dice_after),
                  float(std_dice_after),
                  str(dice_list_after),
                  float(avg_iou_before),
                  float(std_iou_before),
                  str(iou_list_before),
                  float(avg_iou_after),
                  float(std_iou_after),
                  str(iou_list_after),
                  float(avg_hd_before),
                  float(std_hd_before),
                  str(hd_list_before),
                  float(avg_hd_after),
                  float(std_hd_after),
                  str(hd_list_after)
                  ]
        print('Inserting new record...')
        sql = """INSERT INTO {0}(RUN_ID, PERC, SPLIT, CONFIG, EXPERIMENT_TYPE, DATASET_NAME, INPUT_SIZE, EPOCH, 
        TEST_ADAPTATION_LOSS, TEST_ADAPTATION_THRESHOLD, TEST_AUGMENTATION, N_TEST_STEPS_MAX, N_TEST_STEPS,
        ADAPTATION_LOSS_LIST_BEFORE, ADAPTATION_LOSS_LIST_AFTER, ENTROPY_LIST_BEFORE, ENTROPY_LIST_AFTER, 
        AVG_DICE_BEFORE, STD_DICE_BEFORE, DICE_LIST_BEFORE,
        AVG_DICE_AFTER, STD_DICE_AFTER, DICE_LIST_AFTER,
        AVG_IOU_BEFORE, STD_IOU_BEFORE, IOU_LIST_BEFORE,
        AVG_IOU_AFTER, STD_IOU_AFTER, IOU_LIST_AFTER,
        AVG_HD_BEFORE, STD_HD_BEFORE, HD_LIST_BEFORE,
        AVG_HD_AFTER, STD_HD_AFTER, HD_LIST_AFTER, TIMESTAMP) 
        VALUES({1}?)""".format(table_name, '?, ' * (len(values)))
        time_stamp = datetime.datetime.now().strftime("%Y-%b-%d, %A %I:%M:%S")
        values = values + [time_stamp]
        cur = conn.cursor()
        cur.execute(sql, values)

    print('Done.')
