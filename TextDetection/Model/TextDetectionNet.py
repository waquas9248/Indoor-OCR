import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops.boxes as box_ops
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchviz import make_dot

class TextDetectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.concat_conv = nn.Conv2d(64+128, 256, 1)
        self.upconv = nn.ConvTranspose2d(256, 128, 2, stride=2)

        # Anchor Generator
        self.anchor_generator = AnchorGenerator(
            sizes=((16, 32, 64),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        self.detection_head = nn.Conv2d(128, 9*5, 1)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = self.pool1(x1)

        x2 = F.relu(self.bn2(self.conv2(x1)))
        x2 = self.pool2(x2)

        x_fused = torch.cat([
            F.adaptive_avg_pool2d(x1, x2.shape[2:]),
            x2
        ], dim=1)
        x_fused = F.relu(self.concat_conv(x_fused))
        x_up = F.relu(self.upconv(x_fused))

        # Generate anchors
        image_list = ImageList(x, [x.shape[-2:]])
        anchors = self.anchor_generator(image_list, [x_up])[0]

        # Generate predictions
        predictions = self.detection_head(x_up)
        predictions = predictions.permute(0, 2, 3, 1).reshape(
            x_up.size(0), -1, 5
        )

        return anchors, predictions


# Visualization for TextDetectionNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

detection_model = TextDetectionNet().to(device)

dummy_input = torch.randn(1, 1, 1024, 1024).to(device) 

anchors, predictions = detection_model(dummy_input)

dot = make_dot(predictions,
              params=dict(list(detection_model.named_parameters()) + [('input', dummy_input)]),
              )

dot.render("TextDetectionNet_Architecture", format="png")
