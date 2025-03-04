import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import box_iou
from tqdm import tqdm

model = TextDetectionNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

bbox_loss_fn = nn.SmoothL1Loss(reduction='sum') 
confidence_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

def calculate_loss(anchors, preds, targets):
    batch_loss = 0
    batch_size = targets.shape[0]
    
    for i in range(batch_size):
        gt_boxes = targets[i][targets[i].sum(dim=1) > 0]
        if gt_boxes.size(0) == 0:
            continue  
            
        anchor_boxes = anchors[i].clone()
        pred_offsets = preds[i][:, :4]
        pred_conf = preds[i][:, 4]
        
        pred_boxes = torch.zeros_like(anchor_boxes)
        pred_boxes[:, 0] = anchor_boxes[:, 0] + pred_offsets[:, 0] * anchor_boxes[:, 2]
        pred_boxes[:, 1] = anchor_boxes[:, 1] + pred_offsets[:, 1] * anchor_boxes[:, 3]
        pred_boxes[:, 2] = anchor_boxes[:, 2] * torch.exp(pred_offsets[:, 2])
        pred_boxes[:, 3] = anchor_boxes[:, 3] * torch.exp(pred_offsets[:, 3])
        
        iou_matrix = box_iou(pred_boxes, gt_boxes)
        max_iou, gt_indices = iou_matrix.max(dim=1)

        #adjustable
        pos_mask = max_iou > 0.5
        neg_mask = max_iou < 0.3
        
        if pos_mask.sum() > 0:
            matched_gt = gt_boxes[gt_indices[pos_mask]]
            pred_pos = pred_offsets[pos_mask]
            
            target_dx = (matched_gt[:, 0] - anchor_boxes[pos_mask, 0]) / anchor_boxes[pos_mask, 2]
            target_dy = (matched_gt[:, 1] - anchor_boxes[pos_mask, 1]) / anchor_boxes[pos_mask, 3]
            target_dw = torch.log(matched_gt[:, 2] / anchor_boxes[pos_mask, 2])
            target_dh = torch.log(matched_gt[:, 3] / anchor_boxes[pos_mask, 3])
            
            target_offsets = torch.stack([target_dx, target_dy, target_dw, target_dh], dim=1)
            bbox_loss = bbox_loss_fn(pred_pos, target_offsets)
            batch_loss += bbox_loss / pos_mask.sum()  
            
        conf_target = torch.zeros_like(pred_conf)
        conf_target[pos_mask] = 1.0
        conf_loss = confidence_loss_fn(pred_conf, conf_target)
        batch_loss += conf_loss / (pos_mask.sum() + neg_mask.sum())
        
    return batch_loss / batch_size

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    epoch_iterator = tqdm(data_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch")

    for images, targets in epoch_iterator:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        anchors, predictions = model(images)
        
        loss = calculate_loss(anchors, predictions, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0) 
        optimizer.step()

        running_loss += loss.item()
        epoch_iterator.set_postfix(loss=loss.item())

    avg_epoch_loss = running_loss / len(data_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'anchor_sizes': model.anchor_generator.sizes,
        'aspect_ratios': model.anchor_generator.aspect_ratios
    }, final_model_path)
