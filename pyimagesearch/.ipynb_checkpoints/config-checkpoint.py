# import the necessary packages
import torch
import os
# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 3 #og was 1 but since we will be using rgb we use 3 ??????????????
NUM_CLASSES = 35 
NUM_LEVELS = 3
# initialize learning rate, number of epochs to train for, and the
# batch size and also weight decay
INIT_LR = 0.001
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 150 #150 #OG WAS 40
BATCH_SIZE = 1

#define patience and factor for scheduler
SCHED_PATIENCE = 10
SCHED_FACTOR = 0.1

# define the input image dimensions
INPUT_IMAGE_WIDTH = 512
INPUT_IMAGE_HEIGHT = 512
# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "YOUR-MODEL-PATH2")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot2.svg"])
