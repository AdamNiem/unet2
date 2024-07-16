# import the necessary packages
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets
from torchvision.datasets import Cityscapes
import os
from PIL import Image 

class SegmentationDataset(Dataset):
    def __init__(self, root, split, transforms=None):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.root = root
        
        self.transforms = transforms
        
        #Want to get directory of masks and images
        self.images_dir = os.path.join(self.root, "leftImg8bit", split)
        self.masks_dir = os.path.join(self.root, "gtFine", split)
        
        #split is included b/c the data has a train, val, and test folder 
        #ex: urRootPath/gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png  (mask in train folder)
        #ex: urRootPath/leftImg8bit/val/frankfurt/frankfurt_000000_001016_leftImg8bit.png  (img in val folder)
        
        #stores the image paths and mask paths (duh)
        self.image_paths = []
        self.mask_paths = []
        
        #The images are separated in different folders by city so we need to loop through those
        #both images and masks have the same cities (b/c image must have a corresponding img) so we only loop through that once
        for city_name in os.listdir(self.images_dir):
            city_images_path = os.path.join(self.images_dir, city_name)
            city_masks_path = os.path.join(self.masks_dir, city_name)
            
            #loop through the images directory to add the image paths in each city folder to image_paths
            #since each image has a corresponding mask we also get the path of that image's mask 
            for image_name in os.listdir(city_images_path):
                if image_name.endswith("_leftImg8bit.png"): #only want this type of img not the other types in the dir
                    #gets image path
                    image_path = os.path.join(city_images_path, image_name)
                    self.image_paths.append(image_path)
                                        
                    #gets the path of that image's mask
                    #we can do this since .replace() doesn't modify original string
                    mask_name = image_name.replace( "_leftImg8bit.png", "_gtFine_labelTrainIds.png") 
                    mask_path = os.path.join(city_masks_path, mask_name)
                    self.mask_paths.append(mask_path)
                    
    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.image_paths)
    def __getitem__(self, idx):
        # grab the image path from the current index
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        #Disregard these outdated comments
        # load the image from disk, swap its channels from BGR to RGB, (This because cv2 by defaults stores colors as BGR)
        # and read the associated mask from disk in grayscale mode
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        
        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            image, mask = self.transforms(image, mask)        
        # return a tuple of the image and its mask
        return (image, mask)

'''
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class CityscapesDataset(Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        """
        Args:
            root (string): Root directory of the Cityscapes dataset.
            split (string): The dataset split, either 'train', 'val' or 'test'.
            transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
            target_transform (callable, optional): A function/transform that takes in the target mask and transforms it.
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # Images directory
        self.images_dir = os.path.join(root, 'leftImg8bit', split)
        self.images = []
        self.targets = []

        # Walk through the directory and collect all image and target file paths
        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(root, 'gtFine', split, city)
            
            for file_name in os.listdir(img_dir):
                if file_name.endswith('_leftImg8bit.png'):
                    self.images.append(os.path.join(img_dir, file_name))
                    target_name = file_name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                    self.targets.append(os.path.join(target_dir, target_name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        target_path = self.targets[idx]

        img = Image.open(img_path).convert('RGB')
        target = Image.open(target_path)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target

# Example usage
if __name__ == '__main__':
    root_dir = '/path/to/cityscapes'
    transform = T.Compose([
        T.Resize((256, 512)),
        T.ToTensor(),
    ])
    target_transform = T.Compose([
        T.Resize((256, 512), interpolation=Image.NEAREST),
        T.ToTensor(),
    ])
    
    dataset = CityscapesDataset(root=root_dir, split='train', transform=transform, target_transform=target_transform)
    
    # Get a sample from the dataset
    img, target = dataset[0]
    print(f'Image shape: {img.shape}, Target shape: {target.shape}')
'''