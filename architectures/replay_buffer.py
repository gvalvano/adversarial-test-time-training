import numpy as np
import os
import cv2
from idas.utils import safe_mkdir
from glob import glob


class ReplayBuffer(object):
    """Abstract base class for TF-Agents replay buffer."""

    def __init__(self, data_spec, capacity, patience, prioritized_replay=False, buffer_name='replay_buffer',
                 name='replay', **kwargs):
        """Initializes the replay buffer.

        Args:
          data_spec: Specs describing a single item that can be stored in this buffer. Must contain at least input_size
          capacity: number of elements that the replay buffer can hold.
          patience: number of epochs to wait before using replay
          buffer_name: name of the pickle file that will contain the buffer
        """
        self.data_spec = data_spec
        self.capacity = capacity
        self.patience = patience
        self.prioritized_replay = prioritized_replay
        if self.prioritized_replay:
            self.prioritized_condition = 'exponential'
            self.prioritized_tau = np.float32(kwargs['prioritized_tau'])

        self.ds_name = buffer_name
        safe_mkdir(self.ds_name)

        self.buffer_is_empty = True
        self.replay_name = name

    def add_batch(self, fake_items, step='', **kwargs):
        """Adds a batch of items to the replay buffer.
        step can be both an int and a string
        """
        add_condition_true = True
        if self.prioritized_replay:
            replay_step = np.float32(kwargs['replay_step'])
            add_probability = np.exp(- replay_step/self.prioritized_tau)
            add_condition_true = np.random.uniform(0.0, 1.0) < add_probability

        if self.buffer_is_empty or add_condition_true:
            # add elements to the buffer

            files = glob(self.ds_name + '/*.npy')
            if self.capacity <= len(files):
                # reached maximum capacity --> remove older files:
                f_name = np.random.choice(files)
                os.remove(f_name)

            # add element to the buffer:
            num = len([el for el in files if el.rsplit('/')[-1].startswith(step)])  # counter
            f_name = os.path.join(self.ds_name, '{0}_{1}_{2}.npy'.format(step, self.replay_name, num))
            np.save(f_name, fake_items.astype(self.data_spec['dtype']))

            self.buffer_is_empty = False

    def sample_batch(self):
        path = np.random.choice(glob(self.ds_name + '/*_{0}_*.npy'.format(self.replay_name)))
        x = np.load(path).astype(np.float32)
        x = self._py_augment(x)
        return x

    def clear(self):
        """ remove the buffer """
        os.remove(self.ds_name)

    @staticmethod
    def _py_augment(x_batch):
        """ Data augmentation pipeline: add roto-translations """

        n_samples, rows, cols, channels = x_batch.shape
        center = (cols // 2, rows // 2)  # open cv requires swapped rows and cols

        # create and apply transformation matrix for each element
        x_batch_augmented = []
        for i in range(n_samples):
            # sample transformation parameters for the current element of the batch
            minval = - 0.1 * x_batch[i, ..., 0].shape[0]
            maxval = 0.1 * x_batch[i, ..., 0].shape[0]
            tx, ty = np.random.randint(low=minval, high=maxval, size=2)
            scale = np.random.uniform(low=0.98, high=1.02)
            angle = np.random.uniform(low=-90.0, high=90.0)

            class_stack = []
            for c in range(channels):
                curr_slice = x_batch[i, ..., c]
                # transformation matrices:
                m1 = np.float32([[1, 0, tx], [0, 1, ty]])
                m2 = cv2.getRotationMatrix2D(center, angle, scale)

                # apply transformation
                transform_slice = cv2.warpAffine(curr_slice, m1, (cols, rows))
                transform_slice = cv2.warpAffine(transform_slice, m2, (cols, rows))
                class_stack.append(transform_slice)

            x_batch_augmented.append(np.stack(class_stack, axis=-1))

        return np.array(x_batch_augmented)
