# MACS: Multi Domain Adaptation Facilitates Accurate Connectomics Segmentation

This repository contains the implementation of **MACS**, a deep learning framework for connectomics segmentation that introduces the first ever multi-domain adaptation method for connectomics and achieves robust and accurate segmentation performance across diverse connectomics EM datasets.



## Installation
```bash
git clone https://github.com/abrarrahmanabir/MACS.git
cd MACS
```

## Dataset
All preprocessed datasets used in this study are publicly available and include the train, validation, and test splits to ensure reproducibility. You can access the full dataset at the following link:
https://drive.google.com/file/d/1-tuV1zWgcsaz9tEUDRpsrZPoEj9e_glZ/view?usp=sharing
 

## Code Structure
`multidomain.py` : This file contains the complete implementation of our proposed approach **MACS**  and the corresponding source code.

`train_single.py` : This script is used for training each individual source model.

`test.py` : This script contains the code for evaluating the trained models.

`run_macs.sh` : This script runs the complete workflow: dataset setup and validation, single-source training, MACS training, and evaluation. It can safely be restarted because completed checkpoints are skipped.

## Complete Workflow

From the repository directory, run:

```bash
bash run_macs.sh
```

`cmd.txt` contains a safe copy-and-edit command reference, including a fully expanded command with all defaults and examples for partial runs.

Frequently changed values are grouped in uppercase at the top of `run_macs.sh` and are also available as command-line options:

```bash
bash run_macs.sh --help
```

For example, this runs a separate one-epoch smoke training for only the `fly` domain without affecting normal checkpoints:

```bash
bash run_macs.sh --stages single --domains fly --single-epochs 1 --batch-size 1 --models-dir models_smoke
```


### Training
The training process is divided into two main stages: training the individual source models and then training the MACS model.

## 1. Train Single Source Models
We have provided a bash script to automate the training for all source domains.

To run, execute the following command in your terminal:

```bash
bash unet_train.sh
```

## 2. Train the MACS Model
To train the MACS model, use the provided bash script.

```bash

bash run_multi.sh
```

All trained models will be saved in the `./models/` directory.

### Evaluation
We have also provided a complete bash script to automate the evaluation process.

To run the evaluation, execute:


```bash

bash test_multi.sh
```

The evaluation results, including all metrics, will be compiled and saved in the `multidomain_all_results.csv` file.


### Model Architecture
![Model Architecture](macs_model.png)
