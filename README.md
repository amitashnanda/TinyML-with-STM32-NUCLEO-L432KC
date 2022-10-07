# TinyML-with-STM32-NUCLEO-L432KC
This repository shows the design of a light weight CNN based deep-learning algorithm that discriminates life-threatening ventricular arrhythmias (reason for sudden cardiac death) from IEGM recordings and deployed on STM32 NUCLEO-L432KC

## Folder Structure

```
TinyML-with-STM32-NUCLEO-L432KC
│   checkpoint_path
└───data_indices
|    |   test_indice.csv
|    └───train_indice.csv 
|
└───src
│   │   final_model.ipynb
|   |   Dataframesplit.ipynb
|   └───model_1.ipynb
|
└───LICENSE.md 
└───README.md
└───run_jupyter_gpu
└───tools.yml
└───results

```

## Prerequisites
The dependencies are listed under tools.yml and are all purely python based. To install them simply run
```
conda env update --file tools.yml
```

## Dataset
The dataset is private.

## Running
For a local installation, make sure you have pip installed and run: 
```
pip install notebook
```
For running jupyter
```
jupyter notebook
```

## Results
<p align = "justify">

</p>
  

![](/results/1.png)
![](/results/2.png)
