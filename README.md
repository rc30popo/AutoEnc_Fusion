# AutoEnc_Fusion
## Overview
Experiment to generate face image from two different face images by using Autoencoder.
By using Autoencoder with it's intermediate layer size is extremly small compared to input layer, expecting intermediate layer extracts highly abstracted characteristics of human face.
Two human's characteristics are ramdonly combined like genetic operation, and combined data is decoded by Autoencoder.

## Software Version
- python 3.6
	- I'm using anaconda3
- chainer 4.3.1
- openCV 3.4.1

## Scripts

- NN.py
  Define Autoencoder network.
- enc_train3.py
  Train Autoencoder.
  - Variables
    - input_dirs
      List of dir names which include human faces to be learned.
	- model_data_file
	  Filename to be used to load and store neural network model data.
    - optim_data_file
	  Filename to be used to load and store optimizer status data.
	- cont_flag
	  Flag to indicate if learning is restarted from last one which are stored in model_data_file and optim_data_file.
	- n_epoch
	  Total number of epochs
	- batch_size
	  Mini batch size
	- s_epoch
	  Number of epochs to save learned model to file periodically.
- autoenc_run.py
  Execute Autoencoder to image files stored on input dirs.
  - enc_model_data_file
    Filename to be used to load neural network model data.
  - input_dirs
    List of input file directories.
  - output_dir
    Directory name to store decoded images.
- fusion_auto_run.py
  Compose two faces by using Autoencoder
  - enc_model_data_file
    Filename to be used to load neural network model data.
  - input_dirs
    List of input file directories.
	At least total two face images should be stored among these directoris.
  - output_dir
    Directory name to store decoded images.
  - gen_num
    How many number of faces are generated from each combination of two faces
