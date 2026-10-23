"""CNN for 20-day chart images (Jiang, Kelly & Xiu 2023 baseline): 3 blocks of 5x3 conv, BN, LeakyReLU, 2x1 max-pool."""
import torch
import torch.nn as nn


class CNN20(nn.Module):
    def __init__(self):
        super().__init__()

        def block(cin, cout, **kw):
            return nn.Sequential(nn.Conv2d(cin, cout, (5, 3), **kw), nn.BatchNorm2d(cout),
                                 nn.LeakyReLU(0.01), nn.MaxPool2d((2, 1)))
        self.conv = nn.Sequential(
            block(1, 64, stride=(3, 1), dilation=(2, 1), padding=(3, 1)),
            block(64, 128, padding=(2, 1)),
            block(128, 256, padding=(2, 1)),
        )
        n = self.conv(torch.zeros(1, 1, 64, 60)).numel()
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(n, 2))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        return self.fc(self.conv(x))


def batches(X, idx, bs):
    for i in range(0, len(idx), bs):
        j = idx[i : i + bs]
        yield j, torch.from_numpy(X[j]).float().unsqueeze(1)
