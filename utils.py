import cmath
from math import atan2, pi
import random
import numpy as np
import cv2
import tensorflow as tf


def tf_gaussian_noise_layer(input_layer, mean, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=mean, stddev=std, dtype=tf.float32)
    return input_layer + noise


@tf.function
def tf_add_noise_boxes(incoming_mask, n_classes, image_size, mask_type, n_boxes=3, probability=None):
    if probability is None:
        probability = {'random': 1.0, 'jigsaw': 1.0, 'zeros': 1.0}
    for p in probability.values():
        assert 0.0 <= p <= 1.0

    if type(mask_type) is not list:
        assert type(mask_type) == str
        mask_type = [mask_type]

    def _py_corrupt(m):
        m = m.numpy()

        jigsaw_op = np.random.choice([True, False], p=[probability['jigsaw'], 1.0 - probability['jigsaw']])
        zeros_op = np.random.choice([True, False], p=[probability['zeros'], 1.0 - probability['zeros']])
        random_op = np.random.choice([True, False], p=[probability['random'], 1.0 - probability['random']])
        if not (jigsaw_op or zeros_op):
            random_op = True

        for _ in range(n_boxes):

            def get_box_params(low, high):
                # random ray
                r = np.random.randint(low=low, high=high) // 2
                # random center of the box:
                mcx = np.random.randint(r + 1, image_size[0] - r - 1)
                mcy = np.random.randint(r + 1, image_size[1] - r - 1)
                return r, mcx, mcy

            if 'random' in mask_type and random_op:
                r, mcx, mcy = get_box_params(low=1, high=3)
                # labels in the box to 0:
                m[mcx - r:mcx + r, mcy - r:mcy + r, :] = 0
                # set the value to random label:
                m[mcx - r:mcx + r, mcy - r:mcy + r, np.random.randint(n_classes)] = 1
            if 'jigsaw' in mask_type and jigsaw_op:
                # random size of the box:
                ll = np.min([image_size[0], image_size[1]]) // 10
                hh = np.min([image_size[0], image_size[1]]) // 5
                r, mcx, mcy = get_box_params(low=ll, high=hh)

                # labels in the box to 0:
                m[mcx - r:mcx + r, mcy - r:mcy + r, :] = 0
                # choose another box in the image from which copy labels to the previous box:
                mcx_src = np.random.randint(r + 1, image_size[0] - r - 1)
                mcy_src = np.random.randint(r + 1, image_size[1] - r - 1)
                m_copy = m.copy()
                m[mcx - r:mcx + r, mcy - r:mcy + r, :] = m_copy[mcx_src - r:mcx_src + r, mcy_src - r:mcy_src + r, :]
            if 'zeros' in mask_type and zeros_op:
                r, mcx, mcy = get_box_params(low=1, high=10)
                # labels in the box to 0:
                m[mcx - r:mcx + r, mcy - r:mcy + r, :] = 0
                # set the labels in this box to zero:
                m[mcx - r:mcx + r, mcy - r:mcy + r, 0] = 1
        return m.astype(np.float32)

    mask = tf.map_fn(lambda m: tf.py_function(_py_corrupt, [m], tf.float32), elems=incoming_mask, parallel_iterations=20)
    mask = tf.cast(mask, tf.float32)
    mask.set_shape([None, image_size[0], image_size[1], n_classes])
    return mask


@tf.function
def tf_corrupt_mask_with_blobs(mask, n_classes, image_size, n_blobs=3):
    size = image_size
    min_radii = [1, 1]  # int(size[0] * 0.01), int(size[1] * 0.01)
    max_radii = [3, 3]  # [3 * el for el in min_radii]
    offset_add = [0, 0]
    offset_remove = [size[0] // 4, size[1] // 4]

    def _py_corrupt(m):
        add_mask = generate_random_blobs(n_blobs=n_blobs, n_classes=n_classes, image_size=size, offset=offset_add,
                                         min_blob_radii=min_radii, max_blob_radii=max_radii)
        remove_mask = generate_random_blobs(n_blobs=n_blobs*n_classes, n_classes=1, image_size=size, offset=offset_remove,
                                            min_blob_radii=min_radii, max_blob_radii=max_radii)
        add_mask = np.argmax(add_mask, axis=-1)
        remove_mask = remove_mask[..., 0]
        m1 = np.argmax(m, axis=-1)
        m1[np.where(add_mask)] = add_mask[np.where(add_mask)]
        m1[np.where(remove_mask)] = 0
        # m1 = one_hot_encode(m1, n_classes)
        return m1.astype(np.float32)

    mask = tf.map_fn(lambda m: tf.py_function(_py_corrupt, [m], tf.float32), elems=mask, parallel_iterations=20)
    mask = tf.one_hot(indices=tf.cast(mask, tf.int32), depth=n_classes)
    mask = tf.cast(mask, tf.float32)
    mask.set_shape([None, image_size[0], image_size[1], n_classes])
    return mask


def generate_random_blobs(n_blobs, n_classes, image_size, offset, min_blob_radii, max_blob_radii=None):

    def _convexHull(points):
        # Graham's scan.
        x_leftmost, y_leftmost = min(points)
        by_theta = [(atan2(x-x_leftmost, y-y_leftmost), x, y) for x, y in points]
        by_theta.sort()
        as_complex = [complex(x, y) for _, x, y in by_theta]
        cvx_hull = as_complex[:2]
        for pt in as_complex[2:]:
            # Perp product.
            while ((pt - cvx_hull[-1]).conjugate() * (cvx_hull[-1] - cvx_hull[-2])).imag < 0:
                cvx_hull.pop()
            cvx_hull.append(pt)
        return [(pt.real, pt.imag) for pt in cvx_hull]

    def _dft(xs):
        return [sum(x * cmath.exp(2j*pi*i*k/len(xs))
                    for i, x in enumerate(xs))
                for k in range(len(xs))]

    def _interpolateSmoothly(xs, N):
        """For each point, add N points."""
        fs = _dft(xs)
        half = (len(xs) + 1) // 2
        fs2 = fs[:half] + [0]*(len(fs)*N) + fs[half:]
        return [x.real / len(xs) for x in _dft(fs2)[::-1]]

    def _filter_allowed(v, v_max):
        return int(max(0, min(v_max - 1, v)))

    width, height = image_size
    delta_x, delta_y = offset
    mask = np.zeros((width, height, n_classes))
    for b in range(n_blobs):
        for c in range(n_classes):

            if max_blob_radii is None:
                blob_radii = min_blob_radii
            else:
                a = min_blob_radii
                b = max_blob_radii
                blob_radii = [(b[i] - a[i]) * np.random.random_sample() + a[i] for i in range(len(min_blob_radii))]
                blob_radii = [int(el) for el in blob_radii]

            x0 = np.random.random_integers(delta_x, width - delta_x)
            y0 = np.random.random_integers(delta_y, height - delta_y)

            pts = [(random.random() + 0.8) * cmath.exp(2j * pi * i / 7) for i in range(7)]
            pts = _convexHull([(pt.real, pt.imag) for pt in pts])
            xs, ys = [_interpolateSmoothly(zs, 10) for zs in zip(*pts)]
            xs = [_filter_allowed(el * blob_radii[0] + x0, width) for el in xs]
            ys = [_filter_allowed(el * blob_radii[1] + y0, height) for el in ys]

            mask[xs, ys, c] = 1
            # mask[..., c] = area_closing(mask[..., c])
            kernel = np.ones((2 * blob_radii[0], 2 * blob_radii[1]))
            mask[..., c] = cv2.morphologyEx(mask[..., c], cv2.MORPH_CLOSE, kernel=kernel)

    return mask


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    size = (224, 224)
    radii = size[0]//100, size[1]//100
    ofs = [size[0]//10, size[1]//10]
    add_mask = generate_random_blobs(n_blobs=5, n_classes=3, image_size=size, offset=ofs, min_blob_radii=radii)
    ofs = [size[0]//4, size[1]//4]
    remove_mask = generate_random_blobs(n_blobs=3, n_classes=3, image_size=size, offset=ofs, min_blob_radii=radii)
    
    plt.figure()
    plt.imshow(add_mask)
    plt.show()
    
    plt.figure()
    plt.imshow(remove_mask)
    plt.show()
