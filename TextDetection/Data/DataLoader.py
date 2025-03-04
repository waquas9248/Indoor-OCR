from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import json
import os
import torch
from torchvision import transforms

class TextDetDataset(Dataset):
    def __init__(self, json_path, image_folder, transform=None, target_size=(1024, 1024), quantization_levels=16):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.image_folder = image_folder
        self.transform = transform
        self.target_size = target_size
        self.quantization_levels = quantization_levels
        self.img_data = self.data["imgs"]
        self.anns_data = self.data["anns"]
        self.img2anns_data = self.data["imgToAnns"]
        self.target_width, self.target_height = target_size


    def __len__(self):
        return len(self.img_data)

    def load_image(self, img_id):
        img_info = self.img_data[img_id]
        file_name = img_info["file_name"].replace('train/', '')
        img_path = os.path.join(self.image_folder, file_name)

        return Image.open(img_path).convert("RGB")

    def preprocess_image(self, image):
        grayscale_image = image.convert("L")

        orig_width, orig_height = grayscale_image.size
        target_width, target_height = self.target_size

        padding_left = (target_width - orig_width) // 2
        padding_top = (target_height - orig_height) // 2

        padded_image = Image.new("L", self.target_size, color=0)
        padded_image.paste(grayscale_image, (padding_left, padding_top))

        image_np = np.array(padded_image)
        quantized_image = np.floor(image_np / (256 / self.quantization_levels)) * (256 / self.quantization_levels)

        return Image.fromarray(quantized_image.astype(np.uint8)), padding_left, padding_top


    def __getitem__(self, idx):
        img_id = list(self.img_data.keys())[idx]
        image = self.load_image(img_id)
        image, padding_left, padding_top = self.preprocess_image(image)

        anns = self.img2anns_data[img_id]
        gt_boxes = []
        
        for ann_id in anns:
            ann_info = self.anns_data[ann_id]
            
            x1, y1, x2, y2 = ann_info["bbox"]
            
            x1_pad = x1 + padding_left
            y1_pad = y1 + padding_top
            x2_pad = x2 + padding_left
            y2_pad = y2 + padding_top

            cx = ((x1_pad + x2_pad) / 2) / self.target_width
            cy = ((y1_pad + y2_pad) / 2) / self.target_height
            width = (x2_pad - x1_pad) / self.target_width
            height = (y2_pad - y1_pad) / self.target_height

            gt_boxes.append([cx, cy, width, height])

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(gt_boxes, dtype=torch.float32)

transform = transforms.Compose([
    transforms.ToTensor()  
])

def collate_fn(batch):
    images, gt_boxes = zip(*batch)
    images = torch.stack(images)
    
    max_boxes = max([len(b) for b in gt_boxes])
    padded_boxes = []
    
    for boxes in gt_boxes:
        if len(boxes) < max_boxes:
            pad = torch.zeros((max_boxes - len(boxes), 4), dtype=torch.float32)
            padded = torch.cat([boxes, pad])
        else:
            padded = boxes
        padded_boxes.append(padded)

    return images, torch.stack(padded_boxes)

#dataset must be instance of TextDetDataset
data_loader = DataLoader(dataset, batch_size=50, shuffle=True, collate_fn=collate_fn)
