from torch.utils.data import Dataset, IterableDataset
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import openslide
import numpy as np
from PIL import Image
import math
import random
from progressbar import progressbar
import cv2

class WhoRegressionDataSet(IterableDataset):
    
    def __init__(self,path_to_wsis,train_keys,annotation_dict,pseudo_epoch_length,transforms,MitoticCount = False,level = 0):
        super(WhoRegressionDataSet).__init__()
        self.transforms = transforms
        # Path where wsi are
        self.path_to_wsis = path_to_wsis
        # List of keys for training
        self.keys = train_keys
        # Dict that holds the rio, and who-grade, file names are keys
        self.annotation_dict = annotation_dict
        
        self.width = 1024
        self.height = 1024
        # Size of 10 HPF
        self.roi_width = 8284
        self.roi_height = 6213
        self.level = level
        self.pseudo_epoch_length = pseudo_epoch_length
        self.per_worker = None
        self.MitoticCount = MitoticCount
        
        self.worker_info = torch.utils.data.get_worker_info()
        self.load_slides_in_memory()
    
    def __iter__(self):
        # Get worker info
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single thread data loading
            return self.get_random_patch(self.pseudo_epoch_length)
        else:
            # Multy thread data loading
            # Split workload
            per_worker = int(math.ceil(self.pseudo_epoch_length / float(worker_info.num_workers)))
            self.per_worker = per_worker
            self.worker_info = worker_info
            return self.get_random_patch(per_worker)
    
    def __len__(self):
        return self.pseudo_epoch_length
     
    def load_slides_in_memory(self):
        
        self.wsi_container = {
            1:dict(),
            2:dict(),
            3:dict(),
        }
        
        
        for key in progressbar(self.keys):
            slide = openslide.open_slide(self.path_to_wsis + key)

            
            self.wsi_container[self.annotation_dict[key]['who_grade']][key] = dict()
            self.wsi_container[self.annotation_dict[key]['who_grade']][key]['slide'] = slide
            self.wsi_container[self.annotation_dict[key]['who_grade']][key]['roi'] = self.annotation_dict[key]['roi']
            self.wsi_container[self.annotation_dict[key]['who_grade']][key]['who_grade'] = self.annotation_dict[key]['who_grade']
            self.wsi_container[self.annotation_dict[key]['who_grade']][key]['mitotic_count'] = self.annotation_dict[key]['mitotic_count']
            self.wsi_container[self.annotation_dict[key]['who_grade']][key]['down_factor'] = slide.level_downsamples[self.level]
            self.wsi_container[self.annotation_dict[key]['who_grade']][key]['level_dimensions'] = slide.level_dimensions[self.level]
            # When the mitotic count is 0, a random roi with 0.9% > tissue in it is selected as roi
            if self.annotation_dict[key]['roi'] == [0, 0, 8284, 6213]:
                top_left_coords_roi = self.get_coords_for_calculation(slide)

                # Top left coords are (y,x) due to cv2 convention
                self.wsi_container[self.annotation_dict[key]['who_grade']][key]['roi'][0] = top_left_coords_roi[1]
                self.wsi_container[self.annotation_dict[key]['who_grade']][key]['roi'][1] = top_left_coords_roi[0]
                self.wsi_container[self.annotation_dict[key]['who_grade']][key]['roi'][2] = top_left_coords_roi[1] + self.roi_width
                self.wsi_container[self.annotation_dict[key]['who_grade']][key]['roi'][3] = top_left_coords_roi[0] + self.roi_height

                
    def get_random_patch(self,length):
        
        # load a random patch of one of the three who grades
        for i in range(length):
            
            # Select a who grade
            grade = np.random.randint(1,4)
            
            # Select a random file from this grade
            key = np.random.choice(list(self.wsi_container[grade].keys()),replace = False)
            
            # Select a random patch from the roi of this file
            x = np.random.randint(self.wsi_container[grade][key]['roi'][0],self.wsi_container[grade][key]['roi'][2] - self.width * (self.level + 1))
            y = np.random.randint(self.wsi_container[grade][key]['roi'][1],self.wsi_container[grade][key]['roi'][3] - self.width * (self.level + 1))
       
            # Load the corresponding image
            img = Image.fromarray(np.array(self.wsi_container[grade][key]['slide'].read_region((x,y),self.level,(self.width,self.height)))[:,:,:3])
            # Bring everything to pytorch format
            img = self.transforms(img)
            # MSELoss benötigt dtype float32
            if self.MitoticCount:
                # Mitotic count mit ausgeben lassen
                mc = self.wsi_container[grade][key]['mitotic_count']
            
                yield img, torch.tensor(mc,dtype = torch.float32).reshape(-1,1), torch.tensor(grade,dtype = torch.float32).reshape(-1,1)
                
            else:
            
                yield img, torch.tensor(grade,dtype = torch.float32).reshape(-1,1)
            
    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __iter__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        if self.MitoticCount:
            images = list()
            mcs = list()
            targets = list()

            for b in batch:
                images.append(b[0])
                mcs.append(b[1])
                targets.append(b[2])

            images = torch.stack(images, dim=0)
            mcs = torch.cat(mcs)
            targets = torch.cat(targets)

            return images, mcs, targets 
            
        
        else:
            images = list()
            targets = list()

            for b in batch:
                images.append(b[0])
                targets.append(b[1])

            images = torch.stack(images, dim=0)
            targets = torch.cat(targets)

            return images, targets 
            
            
    def make_active_map(self,slide):
        downsamples_int = [int(x) for x in slide.level_downsamples]
        if 32 in downsamples_int:
            self.ds = 32
        elif 16 in downsamples_int:
            self.ds = 16
        else:
            self.showMessageBox('Error: Image does not contain downsampling of factor 16 or 32, which is required.')
            return
        level = np.where(np.abs(np.array(slide.level_downsamples)-self.ds)<0.1)[0][0]
        overview = slide.read_region(level=level, location=(0,0), size=slide.level_dimensions[level])
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(overview)[:,:,0:3],cv2.COLOR_BGR2GRAY)

        # Otsu thresholding
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # Dilate
        dil = cv2.dilate(thresh, kernel = np.ones((7,7),np.uint8))

        # Erode --> yields map
        activeMap = cv2.erode(dil, kernel = np.ones((7,7),np.uint8))
        
        return(activeMap)
    
    def get_coords_for_calculation(self,slide):
        
        coords = list()
        x_max, y_max = slide.dimensions[0], slide.dimensions[1]
        
        activeMap = self.make_active_map(slide)
        activeMap = activeMap / activeMap.max()
        top_lef_coords_roi = self.find_fullest_roi(activeMap)
        return(top_lef_coords_roi)

    def find_fullest_roi(self,activeMap):
        x_length = int(np.ceil(float(self.roi_width)/self.ds))
        y_length = int(np.ceil(float(self.roi_height)/self.ds))
        kernel = np.ones((x_length,y_length), np.float32)
        tissue_map = cv2.filter2D(activeMap, -1, kernel, anchor = (0,0), borderType = cv2.BORDER_CONSTANT)
        roi_center = np.unravel_index(tissue_map.argmax(), tissue_map.shape)
        roi_center = [i * self.ds for i in roi_center]
        return roi_center