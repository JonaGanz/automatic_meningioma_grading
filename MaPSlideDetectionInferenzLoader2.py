from torch.utils.data import Dataset, IterableDataset
import openslide
import numpy as np
from pathlib import Path
from numpy import random
import numpy as np
from itertools import cycle
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import cv2

class MyMapSlideDetectionDataset(Dataset):
    
    def __init__(self,slide,transforms,width = 1024, height = 1024, overlap = 0.1):
        
        self.slide = slide
        self.transforms = transforms
        # Path which holds wsi Images
        self.width = width
        self.height = height
        self.level = 0
        self.overlap = overlap
        
        self.uid_finished = list()        
        self.wsi_annotations = dict()
        self.level_dimension = self.slide.level_dimensions[self.level]
        self.down_factor = self.slide.level_downsamples[self.level]
        
        # Get the coordinates where to infere
        self.coords = self.get_coords_for_calculation()

    
    def __getitem__(self,index):
        
        
        # Make dummy targets so we can use the pytorch dataloader, there migth be a better way to do this
        class_labels = torch.tensor(list(), dtype = torch.int64)
        boxes = torch.tensor(list(), dtype = torch.float32)
        
        coords = self.coords[index]
        image = self.get_patch(coords)
        image = self.transforms(image)
        
        # Return dummy targets
        targets = {
            "boxes":boxes,
            "labels":class_labels,
            "coords":coords
        }
        
        return image, targets
    
    def __next__(self):
        return self.get_box()
    
    def __len__(self):
        # Gives back the pseudo length of the epoch
        return len(self.coords)
                                 
    # Method for geting a patch from Wsi specified by uid at position specified by coords -> (x,y)        
    def get_patch(self,coords):
        return(Image.fromarray(np.array(self.slide.read_region((int(coords[0]),int(coords[1])), self.level, (self.width, self.height)))[:,:,:3]))
        
    
    
    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __iter__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        targets = list()

        for b in batch:
            images.append(b[0])
            targets.append(b[1])
            
        images = torch.stack(images, dim=0)

        return images, targets
    
    def make_active_map(self):
        downsamples_int = [int(x) for x in self.slide.level_downsamples]
        if 32 in downsamples_int:
            self.ds = 32
        elif 16 in downsamples_int:
            self.ds = 16
        else:
            self.showMessageBox('Error: Image does not contain downsampling of factor 16 or 32, which is required.')
            return
        level = np.where(np.abs(np.array(self.slide.level_downsamples)-self.ds)<0.1)[0][0]
        overview = self.slide.read_region(level=level, location=(0,0), size=self.slide.level_dimensions[level])
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(overview)[:,:,0:3],cv2.COLOR_BGR2GRAY)

        # Otsu thresholding
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # Dilate
        dil = cv2.dilate(thresh, kernel = np.ones((7,7),np.uint8))

        # Erode --> yields map
        activeMap = cv2.erode(dil, kernel = np.ones((7,7),np.uint8))
        
        return(activeMap)
    
    def needsCalculation(self,x,y,activeMap):
        x_ds = int(np.floor(float(x)/self.ds))
        y_ds = int(np.floor(float(y)/self.ds))
        step_ds = int(np.ceil(float(self.width)/self.ds))
        needCalculation = np.sum(activeMap[y_ds:y_ds+step_ds,x_ds:x_ds+step_ds])>0.9*step_ds*step_ds
        return needCalculation
    
    def get_coords_for_calculation(self):
        
        coords = list()
        x_max, y_max = self.slide.dimensions[0], self.slide.dimensions[1]
        step_wide = round(self.width - self.overlap * self.width)
        
        activeMap = self.make_active_map()
        
        for x in range(0, x_max, step_wide):
            if x + self.width > x_max:
                x = x_max - self.width
            for y in range(0, y_max, step_wide):
                if y + self.width > y_max:
                    y = y_max - self.height
                if self.needsCalculation(x,y,activeMap):
                    coords.append((x,y))
        return(coords)
            

                        
                        
                 

