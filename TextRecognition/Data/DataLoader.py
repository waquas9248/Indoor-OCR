import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import json
from PIL import Image

class TextOCRDataset(Dataset):
    def __init__(self, json_path, image_folder, transform=None, target_size=(1024, 1024), char_max_len=32):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.image_folder = image_folder
        self.transform = transform
        self.target_size = target_size
        self.img_data = self.data["imgs"]
        self.anns_data = self.data["anns"]
        self.img2anns_data = self.data["imgToAnns"]
        self.char_max_len = char_max_len

    def __len__(self):
        return len(self.img_data)

    def load_and_pad_image(self, img_id):
        img_info = self.img_data[img_id]
        file_name = img_info["file_name"].replace('train/', '') 
        img_path = os.path.join(self.image_folder, file_name)

        image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = image.size
        padded_image = Image.new("RGB", self.target_size, color=(0, 0, 0))
        padding_left = (self.target_size[0] - orig_width) // 2
        padding_top = (self.target_size[1] - orig_height) // 2
        padded_image.paste(image, (padding_left, padding_top))

        return padded_image, padding_left, padding_top

    def adjust_bboxes(self, bbox, padding_left, padding_top):
        x_min, y_min, width, height = bbox
        x_min += padding_left
        y_min += padding_top
        return [x_min, y_min, width, height]

    def __getitem__(self, idx):
        img_id = list(self.img_data.keys())[idx]
        image, padding_left, padding_top = self.load_and_pad_image(img_id)

        anns = self.img2anns_data[img_id]
        samples = []

        for ann_id in anns:
            ann_info = self.anns_data[ann_id]
            bbox = ann_info["bbox"]
            adjusted_bbox = self.adjust_bboxes(bbox, padding_left, padding_top)
            utf8_string = ann_info["utf8_string"]

            x_min, y_min, width, height = map(int, adjusted_bbox)
            bbox_image = image.crop((x_min, y_min, x_min + width, y_min + height))

            if self.transform:
                bbox_image = self.transform(bbox_image)

            ascii_labels = string_to_ascii(utf8_string, self.char_max_len)
            samples.append((bbox_image, ascii_labels))

        return samples

    @staticmethod
    def string_to_ascii(text, max_length=32):
        ascii_ids = [ord(char) if ord(char) < 128 else 0 for char in text[:max_length]]
        padding = [0] * (max_length - len(ascii_ids))
        
        return torch.tensor(ascii_ids + padding, dtype=torch.long)
        
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

def collate_fn(batch):
    images, labels = [], []
    max_width = 128 
    
    for sample in batch:
        for bbox_img, label in sample:
            w, h = bbox_img.size
            new_w = min(max_width, int(w * (32/h))) 
            img = bbox_img.resize((new_w, 32))
            
            padded_img = Image.new('L', (max_width, 32), 0)
            padded_img.paste(img, (0, 0))
            
            images.append(transform(padded_img))
            labels.append(label)
    
    return torch.stack(images), torch.stack(labels)
