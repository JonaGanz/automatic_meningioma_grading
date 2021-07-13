from torch.utils.data import Dataset, IterableDataset
from SlideRunner.dataAccess.database import Database
from SlideRunner.dataAccess.annotations import AnnotationType
import openslide
import numpy as np
import tqdm
from pathlib import Path
from numpy import random
import numpy as np
from itertools import cycle
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class MyIterableSlideDetectionDataset(IterableDataset):
    
    def __init__(self,base_path,uids,database_file,pseudo_epoch_length,transforms,sampling_prob = [0.5,0.5]):
        
        
        self.transforms = transforms
        
        # Path which holds wsi Images
        self.base_path = base_path
        # Sqlite file with annotations
        self.database_file = database_file
        self.uids = uids
        self.slides = []
        self.width = 1024
        self.height = 1024
        self.level = 0
        self.sampling_prob = sampling_prob # [prob mitosis, non mitosis]

        # Since we create a random sample every time we call __init__(), we can basically create an infinite number of different data points. For this reason, a pseudo epoch length is defined, which specifies how many data points should be returned in an epoch. 
        self.pseudo_epoch_length = pseudo_epoch_length

        self.wsi_annotations = dict()
        self.__initialize__()
    
    def __iter__(self):
        return self.get_random_box()
    
    def __next__(self):
        return self.get_random_box()
    
    def __len__(self):
        # Gives back the pseudo length of the epoch
        return self.pseudo_epoch_length
    
    def __initialize__(self):
        
        database = Database()
        database.open(Path.joinpath(self.base_path,self.database_file))
        print(type(database))
        getslides = '''Select uid, filename From Slides'''
        for idx, (currslide, filename) in enumerate(database.execute(getslides).fetchall()):
            if currslide not in self.uids:
                continue

            # load database for each Slide
            database.loadIntoMemory(currslide)
            # open Wsi which have to be located in base_path
            slide = openslide.open_slide(str(Path.joinpath(self.base_path,filename)))
            level_dimension = slide.level_dimensions[self.level]
            down_factor = slide.level_downsamples[self.level]
            
            if currslide not in self.wsi_annotations:
                self.wsi_annotations[currslide] = dict()
                self.wsi_annotations[currslide]['positive'] = list()
                self.wsi_annotations[currslide]['negative'] = list()
                self.wsi_annotations[currslide]['slide'] = slide
                self.wsi_annotations[currslide]['down_factor'] = down_factor
                self.wsi_annotations[currslide]['filename'] = filename
                
            for id, annotation in database.annotations.items():
                if annotation.deleted or annotation.annotationType != AnnotationType.SPOT:
                    continue
                # Calculate coordinates for annotations
                x = annotation.x1 / down_factor
                y = annotation.y1 / down_factor
                
                # Mitosis
                if annotation.agreedClass == 1:
                    # positions for classification task
                    self.wsi_annotations[currslide]['positive'].append((x,y))
                # Non Mitosis
                elif annotation.agreedClass == 2:
                    self.wsi_annotations[currslide]['negative'].append((x,y))
                else:
                    continue
                    
    # Method for geting a patch from Wsi specified by uid at position specified by coords -> (x,y)        
    def get_patch(self,uid,coords):
        # bringing axis shape in Torch format [channels,height,width], casting pil image to array, reading image from wsi
        return(Image.fromarray(np.array(self.wsi_annotations[uid]['slide'].read_region((int(coords[0]-self.width/2),int(coords[1]-self.height/2)), self.level, (self.width, self.height)))[:,:,:3]))
    
    def get_random_sample(self):
        labels = {
            'positive':[1],
            'negative':[0],
        }
        # the pseudo_epoch_length limits the amount of data samples which are given back by the generator
        for i in range(0,self.pseudo_epoch_length):
            # select random wsi
            uid = random.choice(self.uids,1)[0]
            # decide wether a positive or negative sample is selected
            label = random.choice(['positive','negative'],1)[0]
            # select a radom sample from the chosen wsi
            coords = random.choice(np.arange(0,len(self.wsi_annotations[uid][label]),1))
            yield self.transforms(self.get_patch(uid,self.wsi_annotations[uid][label][coords])),labels[label]

    def get_random_box(self):
        
        # Documentation of Resnet Classifier for Pytorch says that 0 is always the background class
        labels = {
            'positive': 1,
        }
        # the pseudo_epoch_length limits the amount of data samples which are given back by the generator
        for i in range(0,self.pseudo_epoch_length):
 
            image, bboxes, class_labels = self.get_image_and_bbox()
            
            difficulties = [0] * len(class_labels)
            image = self.transforms(image)
            class_labels = torch.tensor(class_labels,dtype = torch.int64)
           
            bboxes = torch.as_tensor(bboxes,dtype = torch.float32)
            bboxes = bboxes.reshape(-1, 4)
            # class_labels = class_labels.reshape(-1,1)
            
            # needed for coco evaluaiton
            difficulties = torch.tensor(difficulties, dtype = torch.int64)
            # Define a dictionary to return targets
            targets = {
                "boxes":bboxes,
                "labels":class_labels,
                "difficulties":difficulties
            }


            yield image,targets

    
    
    def get_image_and_bbox(self):
        # select random wsi
        uid = random.choice(self.uids,1)[0]
        # decide wether a positive or negative sample is selected
        label = random.choice(['positive','negative'],replace = False, p = self.sampling_prob)
        # select a radom sample from the chosen wsi
        coords = random.choice(np.arange(0,len(self.wsi_annotations[uid][label]),1))
            
        # random offset for annotation 
        x_offset = random.randint(self.width/2 * -1, self.width / 2)
        y_offset = random.randint(self.height/2 * -1, self.height / 2)
        # get one annotation to sample an image from, these will be the first coordinates in a list with all annotations present in the sample image
        coordinates_list = list()
        class_labels = list()
        anchor_cords = list(self.wsi_annotations[uid][label][coords])
           
        # add random offset to image coordinates
        anchor_cords[0] += x_offset
        anchor_cords[1] += y_offset
            
        # check wether other positive annotations are in the selected patch
        coordinates_list.extend([list(i) for i in self.wsi_annotations[uid]['positive'] if i[0] > anchor_cords[0] - self.width/2 and i[0] < anchor_cords[0] + self.width / 2 and i [1] < anchor_cords[1] + self.height / 2 and i[1] > anchor_cords[1] - self.height/2])
        # coordinates_list.extend([i for i in tmp_list if i != coordinates_list])                    
        class_labels.extend([1] * len(coordinates_list))
            
        # berechnen der bbox
        bboxes = list()
        area = list()
        r = 25
        d = 2 * r / self.wsi_annotations[uid]['down_factor']
            
            
        for xy in coordinates_list:
            # calculate the boxes with respect to the img size for the other samples which might apear in the same patch
            x_min = self.width / 2 + xy[0] - anchor_cords[0] - r
            x_min = 0 if x_min < 0 else x_min
            x_max = x_min + d
            x_max = self.width if x_max > self.width else x_max

            y_min = self.height / 2 + xy[1] - anchor_cords[1] - r
            y_min = 0 if y_min < 0 else y_min
            y_max = y_min + d
            y_max = self.height if y_max > self.height else y_max
            
                   
            bboxes.append([x_min,y_min,x_max,y_max])
            # applying transformations to the image and bboxes
        image = self.get_patch(uid,anchor_cords)
        # image _id needed for coco evaluation

        return image, bboxes, class_labels
    
    
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
        
    
    
    

                        
                        
                 

