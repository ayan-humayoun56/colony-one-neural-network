
import torch
import torch.nn as nn


class TinyColonyNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Small feature extractor
        self.stem = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Keep all 4096 action scores, but reduce head parameters
        self.policy_head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 4096),
        )

        # Predict value from the current player's perspective
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        features = self.stem(x)
        policy = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return policy, value


if __name__ == "__main__":
    model = TinyColonyNet()
    x = torch.zeros(2, 16, 8, 8)
    policy, value = model(x)

    print("Input shape:", x.shape)
    print("Policy shape:", policy.shape)
    print("Value shape:", value.shape)
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))
