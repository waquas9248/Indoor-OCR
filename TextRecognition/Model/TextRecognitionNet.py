import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TextRecognitionNet(nn.Module):
    def __init__(self, num_classes=128, rnn_hidden_size=256, num_rnn_layers=2):
        super().__init__()

        self.glyph_conv = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self._init_glyph_filters()

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.rnn_input_size = 128 * 8
        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=num_rnn_layers,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(rnn_hidden_size * 2, num_classes)

    def _init_glyph_filters(self):
        with torch.no_grad():

            # Vertical Line Filter (Channel 0)
            self.glyph_conv.weight[0] = torch.tensor([
                [[[0,0,1,0,0],
                  [0,0,1,0,0],
                  [0,0,1,0,0],
                  [0,0,1,0,0],
                  [0,0,1,0,0]]]
            ], dtype=torch.float32)

            # Horizontal Line Filter (Channel 1)
            self.glyph_conv.weight[1] = torch.tensor([
                [[[0,0,0,0,0],
                  [0,0,0,0,0],
                  [1,1,1,1,1],
                  [0,0,0,0,0],
                  [0,0,0,0,0]]]
            ], dtype=torch.float32)

            # Left Diagonal Filter (Channel 2)
            self.glyph_conv.weight[2] = torch.tensor([
                [[[1,0,0,0,0],
                  [0,1,0,0,0],
                  [0,0,1,0,0],
                  [0,0,0,1,0],
                  [0,0,0,0,1]]]
            ], dtype=torch.float32)

            # Right Diagonal Filter (Channel 3)
            self.glyph_conv.weight[3] = torch.tensor([
                [[[0,0,0,0,1],
                  [0,0,0,1,0],
                  [0,0,1,0,0],
                  [0,1,0,0,0],
                  [1,0,0,0,0]]]
            ], dtype=torch.float32)

            # Cross Filter (Channel 4)
            self.glyph_conv.weight[4] = torch.tensor([
                [[[0,0,1,0,0],
                  [0,0,1,0,0],
                  [1,1,1,1,1],
                  [0,0,1,0,0],
                  [0,0,1,0,0]]]
            ], dtype=torch.float32)

            # Initialize remaining filters randomly
            nn.init.kaiming_normal_(self.glyph_conv.weight[5:])

    def forward(self, x):
        x = self.glyph_conv(x)

        x = self.conv1(x)
        x = self.conv2(x)

        batch_size, C, H, W = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(batch_size, W, C * H)

        x, _ = self.rnn(x)

        x = self.fc(x)
        return F.log_softmax(x, dim=2)
