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
from idas import utils
from data_interface.utils_acdc.split_data import get_splits
from experiments.base_post_ttt_continual import BaseExperiment
import config as run_config


class Experiment(BaseExperiment):
    def __init__(self, run_id=None, config=None):
        """
        :param run_id: (str) used when we want to load a specific pre-trained model. Default run_id is taken from
                config_file.py
        :param config: arguments from the parser
        """

        self.args = run_config.define_flags() if (config is None) else config

        run_id = self.args.RUN_ID if (run_id is None) else run_id
        assert '&&' in run_id  # '&&' is the divider for run_id
        suffix = '_{0}_{1}'.format(self.args.n_sup_vols, self.args.split_number)
        run_id = run_id.rsplit(suffix)[0]
        self.segmentor_run_id, self.discriminator_run_id = run_id.rsplit('&&')
        self.discriminator_run_id += suffix
        self.segmentor_run_id += suffix
        self.run_id = self.discriminator_run_id + '_TTT'

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.CUDA_VISIBLE_DEVICE)
        self.verbose = self.args.verbose

        if self.verbose:
            print('CUDA_VISIBLE_DEVICE: \033[94m{0}\033[0m\n'.format(str(self.args.CUDA_VISIBLE_DEVICE)))
            print('RUN_ID = \033[94m{0}\033[0m'.format(self.run_id))

        self.num_threads = self.args.num_threads

        # -----------------------------
        # Data

        # data specifics
        self.input_size = (224, 224)
        self.n_classes = 4
        self.batch_size = self.args.batch_size

        # Get the list of data paths
        self.data_path = self.args.data_path
        if self.verbose:
            print('Dataset dir: \033[94m{0}\033[0m\n'.format(self.data_path))

        # get volume ids for train, validation and test sets:
        self.data_ids = get_splits()[self.args.n_sup_vols][self.args.split_number]

        # data pre-processing
        self.augment = self.args.augment  # perform data augmentation
        self.standardize = self.args.standardize  # perform data standardization

        # path to save checkpoints and graph
        results_dir = self.args.results_dir
        utils.safe_mkdir('{0}/results'.format(results_dir))
        self.checkpoint_dir = '{0}/results/{1}/checkpoints/'.format(results_dir, self.args.dataset_name)
        self.checkpoint_segmentor = self.checkpoint_dir + self.segmentor_run_id
        self.checkpoint_discriminator = self.checkpoint_dir + self.discriminator_run_id

        # call super init
        super().__init__(run_id=run_id, config=self.args)

    def get_data(self):
        """ Define the dataset iterators for each task (supervised, unsupervised, mask discriminator, future prediction)
        They will be used in define_model().
        """

        self.global_seed = tf.placeholder(tf.int64, shape=())

        _, self.valid_init, self.test_init, self.input_images, self.output_masks = \
            super(Experiment, self).get_acdc_sup_data(data_path=self.data_path, data_ids=self.data_ids,
                                                      repeat=False, seed=self.global_seed)


if __name__ == '__main__':
    print('\n' + '-' * 3)
    model = Experiment()
    model.build()
    model.test()
