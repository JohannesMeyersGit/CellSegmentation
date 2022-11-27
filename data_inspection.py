import numpy as np
import pandas as pd 
from skimage import io
import glob as glob
import json
import os



dir_to_data = r'D:\Code\Dataset\LIVECell_dataset_2021\images\images\livecell_test_images'
dir_to_labels = r'D:\Code\Dataset\LIVECell_dataset_2021\annotations\LIVECell\livecell_coco_train.json'


def get_image_id_by_name(name, coco_labels):
    for i in range(len(coco_labels['images'])):
        print(coco_labels['images'][i]['file_name'])
        if coco_labels['images'][i]['file_name'] == name:
            id = coco_labels['images'][i]['id']
            
            return id
    
    
    return ''


# load image file paths
ims = glob.glob(dir_to_data + '\*.tif')

# load label json object 

with open(dir_to_labels) as f:
    coco_labels = json.load(f)

imname = os.path.split(ims[0])[-1]
im = io.imread(ims[0])
#io.imshow(im)
#io.show()

single_label = get_image_id_by_name(imname, coco_labels)


print(single_label)
