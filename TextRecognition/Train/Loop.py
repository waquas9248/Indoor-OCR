import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from ../model/TextRecognitionNet import TextRecognitionNet

import json
from tqdm import tqdm


def text_to_ascii(text):
    return [ord(c) if ord(c) < 128 else 0 for c in text]

ascii_mapping = {i: chr(i) for i in range(128)}
with open("ascii_mapping.json", "w") as f:
    json.dump(ascii_mapping, f)
    

criterion = nn.CTCLoss(blank=0)  
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize model with fixed 128 classes
model = TextRecognitionNet(num_classes=128).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CTCLoss(blank=0)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(data_loader):
        images = images.to(device)
        
        targets = [torch.tensor(text_to_ascii(label), dtype=torch.long) for label in labels]
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
        
        logits = model(images)
        
        input_lengths = torch.full((images.size(0),), logits.size(1), dtype=torch.long)
        target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
        
        loss = criterion(
            logits.permute(1, 0, 2),  
            targets.to(device),
            input_lengths,
            target_lengths.to(device)
        )
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        
        running_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {running_loss/len(data_loader):.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, "text_recognition_model.pth")
