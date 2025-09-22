import os
import json
import torch
from torch.utils.data import Dataset

from PIL import Image


class ObjectDetectionDataset(Dataset):
    def __init__(self, root_dir, transform=None, test=True):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.annotation_files = []
        self.test = test

        # Populate image and annotation file lists
        print(root_dir)
        for filename in os.listdir(root_dir):
            if not filename.endswith(".json") and not self.test:
                continue

            root_filename = filename.split(".")[0]

            image_file = os.path.join(root_dir, root_filename + ".jpg")
            annotation_file = os.path.join(root_dir, filename)

            if os.path.exists(image_file):
                self.image_files.append(image_file)
                self.annotation_files.append(annotation_file)

    def _read_and_validate_annotation_file(json_file):
        with open(json_file, "r") as file:
            annotation_file = json.load(file)

        shapes = annotation_file["shapes"]

        assert len(shapes) == 2, "more than 2 labels provided"
        assert "0" in [x["label"] for x in shapes], "no grid label provided"
        assert "1" in [x["label"] for x in shapes], "no cross hair label provided"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')
        initial_size = img.size

        annotation_path = self.annotation_files[idx]
        boxes = []
        labels = []
        filename = img_path.split('/')[-1]
        if not self.test:
            with open(annotation_path, 'r') as file:
                annotations = json.load(file)

            for line in annotations['shapes']:
                
                boxes.append(line['points'])
                labels.append(int(line['label']))

            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            
            cross_hair_box = boxes[1]

            x1 = cross_hair_box[0,0]
            y1 = cross_hair_box[0,1]
            x2 = cross_hair_box[1,0]
            y2 = cross_hair_box[1,1]

            x_box = torch.tensor(((x1, 0), (x2, img.size[1])))
            y_box = torch.tensor(((0, y1), (img.size[0], y2)))

            boxes = torch.stack((boxes[0], x_box, y_box))
            labels = torch.concat((labels, torch.tensor((2,))))
        
        if self.transform:
            
            img = self.transform(img)            
            resized_shape = img.size()
            scale_factor = initial_size[0]/resized_shape[2]
            
            if len(boxes)!=0:
                boxes /= scale_factor
        
        return img, boxes, labels, filename