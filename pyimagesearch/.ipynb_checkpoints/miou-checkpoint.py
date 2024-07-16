import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import numpy as np
from pyimagesearch import config
from torchvision.transforms import InterpolationMode 


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, val: torch.Tensor or int or float, delta_n=1):
        self.count += delta_n
        self.sum += val * delta_n

    def get_count(self) -> torch.Tensor or int or float:
        return self.count.item() if isinstance(self.count, torch.Tensor) and self.count.numel() == 1 else self.count

    @property
    def avg(self):
        avg = -1 if self.count == 0 else self.sum / self.count
        return avg.item() if isinstance(avg, torch.Tensor) and avg.numel() == 1 else avg


# some datasets have an "ignore index", an null class essentially in the data, not to be used for calculation.
# this can be set if needed
class SegIoU:
    def __init__(self, num_classes: int, ignore_index: int = 4) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = (outputs + 1) * (targets != self.ignore_index)
        targets = (targets + 1) * (targets != self.ignore_index)
        intersections = outputs * (outputs == targets)

        outputs = torch.histc(
            outputs,
            bins=self.num_classes,
            min=1,
            max=self.num_classes,
        )
        targets = torch.histc(
            targets,
            bins=self.num_classes,
            min=1,
            max=self.num_classes,
        )
        intersections = torch.histc(
            intersections,
            bins=self.num_classes,
            min=1,
            max=self.num_classes,
        )
        unions = outputs + targets - intersections

        return {
            "i": intersections,
            "u": unions,
        }

# val loader if your validation dataset dataloader
def eval_IoU(model, val_loader):

    model.eval()
    interaction = AverageMeter()
    union = AverageMeter()
    
    iou = SegIoU(num_classes=4, ignore_index=4)
    
    toTensor = transforms.ToTensor() #normalizes the input images

    with torch.inference_mode():
        for idx, batch in enumerate(val_loader):
            #images, mask = feed_dict["data"].cuda(), feed_dict["label"].cuda()
            
            origMask = Image.open(val_loader.dataset.mask_paths[idx])
            origMaskWidth, origMaskHeight = origMask.size #returns tuple in format (width, height) 
            origMaskNpArr = np.array(origMask)
            mask = torch.tensor(origMaskNpArr, dtype=torch.int64).to(config.DEVICE)

            #We want the original image without the data aug applied to it and as a tensor stored on the gpu since the model is on the gpu
            origImg = Image.open(val_loader.dataset.image_paths[idx])
            origImgTensor = toTensor(origImg).to(config.DEVICE)

            #add a batch dimension of size 1 to the origImgTensor so that it works with the model
            origImgTensor = origImgTensor.unsqueeze(0)
            
            # compute output
            output = model(origImgTensor)
            
            # resize the output to match the shape of the mask
            resize = transforms.Resize((origMaskHeight, origMaskWidth), interpolation=InterpolationMode.NEAREST, antialias=False) #pytorch expects size to be passed as (h, w)
            output = resize(output) #resize the predicted mask tensor
            
            output = torch.argmax(output, dim=1)
            
            stats = iou(output, mask)
            interaction.update(stats["i"])
            union.update(stats["u"])
                 
    return (interaction.sum / union.sum).cpu().mean().item()

