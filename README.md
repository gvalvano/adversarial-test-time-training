  
#  Stop Throwing Away Discriminators! <br>Re-using Adversaries for Test-Time Training  
  Code for the paper:    

> Valvano G., Leo A. and Tsaftaris S. A. (DART, 2021), *Stop Throwing Away Discriminators! Re-using Adversaries for Test-Time Training*.    
 The official project page is [here](https://vios-s.github.io/adversarial-test-time-training/).    
An online version of the paper can be found [here](https://arxiv.org/abs/2108.12280).    
  
## Citation: 
``` 
@incollection{valvano2021selfsup,  
 title={Stop Throwing Away Discriminators! Re-using Adversaries for Test-Time Training},      author={Gabriele Valvano and Andrea Leo and Sotirios A. Tsaftaris},  
 year={2021}, booktitle={Domain Adaptation and Representation Transfer},}  
```    
 <img src="https://github.com/vios-s/adversarial-test-time-training/blob/main/images/banner.png" alt="adversarial_ttt" width="800"/>  
  
----------------------------------    
 ## Notes:   
For the experiments, refer to the files:
- **`experiments/base_gan_ttt.py`**. This file contains the model and all the code needed for training. It is the base class inherited from the class `Experiment()` inside `experiments/acdc/exp_gan_ttt.py`. Refer to the class method `define_model()` to see how we build the CNN architectures.  The structure of segmentor, discriminator, and adaptor can be found under the folder `architectures`.  
- **`experiments/acdc/exp_gan_ttt.py`**. This file defines a child class inheriting from the base class defined in `experiments/base_gan_ttt.py`. It defines the directories and filenames needed for the logs, and also the `get_data()` method, which wraps the experiment to the dataset used for the experiments.
    
Once you download the [ACDC dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html), you can pre-process it using the code in the file `data_interface/utils_acdc/prepare_dataset.py`. 
You can also **train with custom datasets**, but you must adhere to the template required by `data_interface/interfaces/dataset_wrapper.py`, which assumes the access to the dataset is through a tensorflow dataset iterator. Moreover, you will need to modify the method `get_data()` inside `experiments/acdc/exp_gan_ttt.py`.

You can **start training** following the guidelines in `run.sh`. To run the training on GPU #0 you can type in the shell: `sh run.sh 0` where 0 is the GPU number. The training will proceed for both experiments in:
- semi-supervised learning (Non-Identifiable Distribution Shift between train and test set), splitting the dataset in 40-20-40% of samples for train, validation and test sets (training annotations only for 25% of the training data);
- training on 1.5T MRI scanners and testing on 3T scanners (Identifiable Distribution Shift).
After training, the script also performs the test using **Adversarial Test-Time Training** in its standard formulation, and in a continual learning setting.

After you run the script, you can **monitor the training process** using tensorboard:  
`tensorboard --logdir=results/acdc/graphs/`  
and then using your browser to navigate to the returned http address (defaults on localhost:6006). 
  
## Requirements  
This code was implemented using TensorFlow 1.14 and the libraries detailed in [requirements.txt](https://github.com/gvalvano/multiscale-pyag/requirements.txt).  
You can install these libraries as:  
`pip install -r requirements.txt`  
or using conda (see [this](https://stackoverflow.com/questions/51042589/conda-version-pip-install-r-requirements-txt-target-lib)).  
  
We tested the code on a TITAN Xp GPU, and on a GeForce GTX 1080, using CUDA 10.2.