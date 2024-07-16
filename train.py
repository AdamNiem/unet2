#notes i think the old images did not work as it was a grayscale image where each different gray color was a float like 0.26 which is not good for crossentropyloss since it requires integers

#from torchvision import transforms
from torch.nn import CrossEntropyLoss #we will use crossEntropyLoss here since we have multiple classes
from torch.optim import Adam
import torchvision.transforms as transforms
from torchvision.transforms import v2
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
from PIL import Image

# import the necessary packages
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets
import os
from PIL import Image 
import numpy as np

import json

# USAGE
# python train.py
# import the necessary packages
#from pyimagesearch.dataset2 import SegmentationDataset
from pyimagesearch.model3 import UNet
from pyimagesearch import config
from pyimagesearch.miou import eval_IoU
from torch.utils.data import DataLoader


class SegmentationDataset(Dataset):
    def __init__(self, root, split, transforms=None):
        #loop through each location
        #Go into the split folder
        #Go into images and masks since they will be in pairs

        # store the image and mask filepaths, and augmentation
        # transforms
        self.root = root

        self.transforms = transforms

        #split is included b/c the data has a train, val, and test folder 
        #ex: urRootPath/Brown_Field/Train/annos/int_maps/mask_101.png  (mask in train folder)
        #ex: urRootPath/ (img in val folder)

        #stores the image paths and mask paths (duh)
        self.image_paths = []
        self.mask_paths = []

        #weird, int_map masks in brown_field is named mask_numberHere.png while in Powerline its anno_pln_numberHere.png
        for trail_name in os.listdir(self.root):
            split_path = os.path.join(root, trail_name, split.capitalize())
            imgs_path = os.path.join(split_path, "imgs")
            masks_path = os.path.join(split_path, "annos/int_maps/")
            for mask_name in os.listdir(masks_path):

                    #Get the int_map mask first
                    mask_path = os.path.join(masks_path, mask_name)
                    self.mask_paths.append(mask_path)

                    #Get the corresponding image to that map_mask
                    image_name = "img_" + mask_name.split("_", 1)[1] #need to change file name a little for images
                    image_path = os.path.join(imgs_path, image_name)
                    self.image_paths.append(image_path)

        #The images are separated in different folders by trail so we need to loop through those
        #both images and masks have the same trail (b/c image must have a corresponding mask) so we only loop through that once

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.image_paths)
    def __getitem__(self, idx):
        # grab the image path from the current index
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            image, mask = self.transforms(image, mask)
        # return a tuple of the image and its mask
        return (image, mask)


from torchvision.transforms import InterpolationMode 

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        '''
        for t in self.transforms:
            torch.manual_seed(42) #the manual seed ensure that both the mask and target have the same parameters passed to the transformation done on the two 
            image = t(image)
            torch.manual_seed(42)
            target = t(target)
        '''
        #Numpy Array stores image data as [Height, Width, Num_Color_Channels]
        #Pytorch Tensor stores image data as [Num_Color_Channels, Height, Width]

        #We cannot use transforms.ToTensor() for masks as it normalizes the data between 0.0 and 1.0 which we dont want
        #So instead we first convert to numpy array and then to tensor of type int 64 to avoid this
        #Thus our integer labels in the image data is preserved

        flip = transforms.RandomHorizontalFlip(p=0.5)

        #mask
        #########################
        randResizeCrop = transforms.RandomResizedCrop(size=(config.INPUT_IMAGE_HEIGHT,config.INPUT_IMAGE_WIDTH), interpolation=InterpolationMode.NEAREST, antialias=False) # Or Resize(antialias=False)
        randRotate = transforms.RandomRotation(10, interpolation=InterpolationMode.NEAREST, fill=4) #the 4 will be specified to be ignored by cross-entropy loss (num has to be from 0-255 to work)

        torch.manual_seed(42)
        target = flip(target) #flip

        #torch.manual_seed(42)
        #target = randResizeCrop(target) #rand resize crop

        torch.manual_seed(42)
        target = randRotate(target) #rotate

        #however we do want to use ToTensor for images to normalize to help prevent giving too high of an initial value to color values

        target = np.array(target)
        target = torch.tensor(target, dtype=torch.int64) #int64 b/c target (in this case masks) have to be int64 for crossEntropyLoss

        #image
        #########################
        #Note, bicubic takes in like 4x the amount of pixels when doing the resize than bilinear so if its so slow maybe make that swap
        randResizeCrop = transforms.RandomResizedCrop(size=(config.INPUT_IMAGE_HEIGHT,config.INPUT_IMAGE_WIDTH), interpolation=InterpolationMode.BILINEAR)  # Or Resize(antialias=True)
        randRotate = transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR)

        torch.manual_seed(42) 
        image = flip(image) #flip

        #torch.manual_seed(42)
        #image = randResizeCrop(image) #rand resize crop

        torch.manual_seed(42)
        image = randRotate(image) #rotate

        colorJitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        image = colorJitter(image)

        toTensor = transforms.ToTensor() #toTensor also does data normalization
        image = toTensor(image)

        return image, target

#WARNING: Do not use Transform.ToTensor as it normalizes the data [0.0-1.0] which we don't want
#Aug apply to mask and img

#NOTES: 
#May try the timm randaug
#
transform = [
    #v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image (supposedly this makes its faster but havent tried it)
    #WARNING: antialias=false is needed to prevent the pixels in mask from being blurred when stretched thus messing up the labelIds
    #transforms.RandomHorizontalFlip(p=0.5),
]

#WARNING: Interpolation nearrest is needed to prevent the pixels in mask from being blurred when stretched thus messing up the labelIds
#transform.append( transforms.Resize((config.INPUT_IMAGE_HEIGHT,config.INPUT_IMAGE_WIDTH), interpolation=Image.NEAREST )) 

transform = Compose(transform)

trainDS = SegmentationDataset('/scratch/aniemcz/CAT', split='Train',
    transforms=transform)
testDS = SegmentationDataset('/scratch/aniemcz/CAT', split='Test',
    transforms=transform)


print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")
# create the training and test data loaders

trainLoader = DataLoader(trainDS, shuffle=True,
    batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)
testLoader = DataLoader(testDS, shuffle=False,
    batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)
testLoaderEval = DataLoader(testDS, shuffle=False,
    batch_size=1, pin_memory=config.PIN_MEMORY)

from torch.optim.lr_scheduler import ReduceLROnPlateau

# initialize our UNet model
unet = UNet(in_channels=3, classes=4).to(config.DEVICE)
# initialize loss function and optimizer
#0 is the index for everything that isn't the traversable terrain
#REMOVED IGNORE_INDEX=0 SINCE DIDNT KNOW HOW TO FIX THIS WITH MIOU FOR NOW
#Should try to see if I instead just do ignore_index = 0 to ignore the forest and background entirely
lossFunc = CrossEntropyLoss(ignore_index=4)
#added weight decay and
opt = Adam(unet.parameters(), lr=config.INIT_LR, weight_decay=config.WEIGHT_DECAY)
#scheduler uses miou to measure (higher miou is better so we use max)
scheduler = ReduceLROnPlateau(opt, mode='max', factor=config.SCHED_FACTOR, patience=config.SCHED_PATIENCE)

# calculate steps per epoch for training and test set
trainSteps = len(trainDS) // config.BATCH_SIZE
testSteps = len(testDS) // config.BATCH_SIZE
# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": [], "accuracies": []}

patience = 10 #If the miou does not increase after ten epochs then kill the run
current_patience = 0

from torch.nn import Softmax
# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
bestMiou = 0.0
for e in range(config.NUM_EPOCHS):
    # set the model in training mode
    unet.train()
    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalTestLoss = 0
    #trainLoader = tqdm(trainLoader)
    # loop over the training set
    for (i, (x, y)) in enumerate(trainLoader):
        # send the input to the device
        (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
        # perform a forward pass and calculate the training loss
        pred = unet(x)
        loss = lossFunc(pred, y)
        # first, zero out any previously accumulated gradients, then
        # perform backpropagation, and then update model parameters
        opt.zero_grad()
        loss.backward()
        opt.step()
        # add the loss to the total training loss so far
        totalTrainLoss += loss

    # switch off of autograd occurs in the calc_miou function and also the e % 5 is so it only takes miou every 5 epochs
    #if (e+1) % 5 == 0 and (e+1) != 0: #calc miou on every 5th epoch
    if (e+1) % 20 == 0:
        torch.save(unet, config.MODEL_PATH + f"_{e}")
        with open("output/data.json", "w") as json_file:
            json.dump(H, json_file, indent=4)
    if True==True: #will calculate miou on every epoch
        #Scheduler to reduce learning rate once the miou isnt improving that much
        miou = eval_IoU(unet, testLoaderEval)
        print(f"miou: {miou}")
        scheduler.step(miou)
        #record miou and the epoch
        H["accuracies"].append({"miou":miou, "epoch": e}) #adds dict containing miou and the epoch
    avgTrainLoss = totalTrainLoss / trainSteps
    print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
    print("Train loss: {:.6f}".format(
        avgTrainLoss))
    
    # Check if validation loss has improved
    if miou > bestMiou:
        bestMiou = miou
        current_patience = 0
    else:
        current_patience += 1
        if current_patience == patience:
            print("Validation loss hasn't improved for {} epochs, stopping training...".format(patience))
            break
# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
    endTime - startTime))


import matplotlib.pyplot as plt

# Extract MIoU and epoch data
mious = [miou["miou"] for miou in H["accuracies"]]
epochs = [epoch["epoch"] for epoch in H["accuracies"]]

# Create the plot
plt.style.use("ggplot")
plt.figure(figsize=(10, 6))  # Adjust figure size if needed

# Plot MIoU against epochs
plt.plot(epochs, mious, label="MIoU", marker='o')

# Set labels and title
plt.title("UNET Performance")
plt.xlabel("Epoch")
plt.ylabel("MIoU")

# Add legend
#plt.legend(loc="best")

# Customize x-axis ticks
plt.xticks(epochs)  # Show every 5th epoch on x-axis, adjust as needed

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Save the figure
plt.savefig(config.PLOT_PATH)
#plt.close()

# Optionally, display the plot
#plt.show()

# Serialize the model to disk
#torch.save(unet.state_dict(), config.MODEL_PATH)
torch.save(unet, config.MODEL_PATH)

with open("output/data.json", "w") as json_file:
    json.dump(H, json_file, indent=4)