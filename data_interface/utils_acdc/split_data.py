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

import numpy as np


def get_splits():
    """""
    Returns an array of splits into validation, test and train indices.
    """

    splits = {

        # this trains/validate on 1.5 T images and tests on patients acquired with a 3T scanner
        '1p5T_to_3T': {
            'split0': {
                'test': [4, 6, 8, 9, 10, 22, 24, 35, 38, 39, 40, 41, 43, 45, 47, 48, 57, 63, 68, 69, 70, 73, 75,
                         77, 81, 82, 83, 84, 86, 87, 88, 90, 99],
                'validation': [76, 28, 78, 89, 98, 29, 58, 95, 55, 2, 1, 15, 25, 67, 31, 60, 52, 100, 93, 54, 33, 96],
                'train_sup': [26, 91, 49, 3, 27, 30, 71, 18, 97, 36, 85, 56, 7, 61, 53, 34, 64, 12, 11, 62, 50, 32],
                'train_disc': [26, 91, 49, 3, 27, 30, 71, 18, 97, 36, 85, 56, 7, 61, 53, 34, 64, 12, 11, 62, 50, 32,
                               44, 79, 59, 17, 13, 37, 92, 14, 51, 94, 5, 74, 23, 19, 20, 80, 46, 21, 66, 72, 16, 42, 65],
                'train_unsup': [44, 79, 59, 17, 13, 37, 92, 14, 51, 94, 5, 74, 23, 19, 20, 80, 46, 21, 66, 72, 16, 42, 65],
            }
        },


        # -----------------
        # 40-20-40

        'perc25': {
            'split0': {'test': [3, 7, 10, 11, 13, 18, 20, 22, 30, 32, 33, 39, 41, 42, 45, 47, 48, 49, 50, 51, 57, 61,
                                67, 68, 69, 70, 71, 76, 77, 79, 80, 82, 83, 86, 88, 91, 92, 94, 96, 97],
                       'validation': [2, 4, 5, 9, 14, 21, 26, 29, 40, 44, 56, 60, 64, 65, 72, 73, 78, 87, 89, 98],
                       'train_sup': [43, 38, 27, 74, 53, 23, 52, 46, 36, 54],
                       'train_disc': [1, 6, 8, 12, 15, 16, 17, 19, 23, 24, 25, 27, 28, 31, 34, 35, 36, 37, 38, 43, 46,
                                      52, 53, 54, 55, 58, 59, 62, 63, 66, 74, 75, 81, 84, 85, 90, 93, 95, 99, 100],
                       'train_unsup': [1, 6, 8, 12, 15, 16, 17, 19, 24, 25, 28, 31, 34, 35, 37, 55, 58, 59, 62, 63, 66,
                                       75, 81, 84, 85, 90, 93, 95, 99, 100]
                       }
        },

        # -----------------
        # All the data:

        '1p5T': [1, 2, 3, 5, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                 34, 36, 37, 42, 44, 46, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 64, 65, 66, 67, 71, 72,
                 74, 76, 78, 79, 80, 85, 89, 91, 92, 93, 94, 95, 96, 97, 98, 100],

        '3T': [4, 6, 8, 9, 10, 22, 24, 35, 38, 39, 40, 41, 43, 45, 47, 48, 57, 63, 68, 69, 70, 73, 75, 77, 81, 82, 83,
               84, 86, 87, 88, 90, 99],

        'all_data': {'all': list(np.arange(100) + 1)}
    }
    return splits


if __name__ == '__main__':
    _splits = get_splits()
    for k, v in zip(_splits.keys(), _splits.values()):
        print('\n' + '- '*20)
        print('number of volumes: {0}'.format(k))
        print('values:', v)
