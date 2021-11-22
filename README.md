# Traffic light recognition
### Requirements
The required packages can be found in *config/env_files/active_learing_env.yml*. 
Dependencies could be installed by running:
> conda env create -f config/env_files/active_learning_env.yml

### Configuration
The experiments are run according to configurations. The config files can be found in 
*config/config_files*.
Configurations can be based on each other (*base_config* key). This way the code will use the parameters of the specified 
base config and only the newly specified parameters will be overwritten.
 
The base config file is *base.yaml*. A hpo example can be found in *base_hpo.yaml*
which is based on *base.yaml* and does hyperparameter optimization only on the specified parameters.

Every experiment has its config file and hpo config file. 

### Arguments
The code should be run with arguments: 

*--id_tag* specifies the name under the config where the results will be saved \
*--config* specifies the config name to use (eg. --config "base" for *config/config_files/base.yaml*)\
*--mode* can be 'train', 'val' or 'hpo' 

### Required data
I used cifar-10 for the experiments which can be downloaded from 
[here](https://www.cs.toronto.edu/~kriz/cifar.html). 
After extracting it the required data folder's path should be specified inside the config file like:
> data: \
  &emsp; params: \
  &emsp;&emsp; dataset_path: '/home/data/cifar-10-batches-py' \

To set which data part (the full training, the labelled, the newly picked ones or their combinations).
state the *data_part* value accordingly in the config path. The usable data parts correspond to the files in
*data/data_parts/{data_part}*.

An example of using the labelled dataset and randomly picked 5000 images from the unlabelled dataset:
> data: \
&emsp; params: \
&emsp;&emsp; data_parts: ['labelled', 'random'] \

### Saving and loading experiment
The save folder for the experiment outputs can be set in the config file like:
> id: "base"\
  env: \
  &emsp; result_dir: 'results'

All the experiment will be saved under the given results dir: {result_dir}/{config_id}/{id_tag arg}
1. tensorboard files
2. train and val metric csv
3. the best model
4. confusion matrices and by class metrics

If the result dir already exists and contains a model file then the experiment will automatically resume
(either resume the training or use the trained model for inference.)

### Usage
##### Training
To train the model use:
> python run.py --config base --mode train

#### Eval
For eval the  results dir ({result_dir}/{config_id}/{id_tag arg}) should contain a model as 
*model_best.pth.tar*. During eval the validation files will be inferenced and the metrics will be calculated.
> python run.py --config base --mode val

#### Save predictions for an eval
To save the predictions for an eval (like saving the predictions of unlabelled data)
an eval should be run with 
1. stating the result dir of the model to use for the prediction 
2. writing *env: save_preds: true* to the config file
3. setting id_tag argument empty: *--id_tag ""*

An example of the required config file can be found in *config/config_files/save_unlabelled_pred_from_labelled.yaml*
> python run.py --config save_unlabelled_pred_from_labelled --mode val --id_tag ""

#### HPO
For hpo use:
> python run.py --config base_hpo --mode hpo

#### Picking files for active learning
To use different active learning methods to pick new images to label run one of the scripts in *active_learning* 
folder. Before running overwrite the pred_path to the path of your train's prediction file 
(refer to:  *Save predictions for an eval*).
These will generate the corresponding files into *data/data_parts/{active_learning_name}*, containing the picked data with a particular active
learning method.

For further information about using the picked data refer to *Required data* 
