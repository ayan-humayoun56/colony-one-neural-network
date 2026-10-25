import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from tiny_colony_net import TinyColonyNet


def main():
    # Make experiments reproducible.
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cpu")
    print(f"Training on: {device}")

    # Load the self-play dataset.
    data = np.load("selfplay_data.npz")

    observations = torch.from_numpy(
        data["observations"].astype(np.float32)
    )
    policies = torch.from_numpy(
        data["policies"].astype(np.float32)
    )
    values = torch.from_numpy(
        data["values"].astype(np.float32)
    )

    print("Observations:", observations.shape)
    print("Policies:", policies.shape)
    print("Values:", values.shape)

    dataset = TensorDataset(observations, policies, values)

    # Keep a small validation subset to check that training runs correctly.
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False
    )

    model = TinyColonyNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Policy targets are probability distributions over 4096 actions.
    # Value targets are -1 (loss) or +1 (win).
    value_loss_fn = nn.MSELoss()

    epochs = 40

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0

        for x, target_policy, target_value in train_loader:
            x = x.to(device)
            target_policy = target_policy.to(device)
            target_value = target_value.to(device)

            predicted_logits, predicted_value = model(x)

            log_probs = torch.log_softmax(predicted_logits, dim=1)
            policy_loss = -(
                target_policy * log_probs
            ).sum(dim=1).mean()

            value_loss = value_loss_fn(
                predicted_value, target_value
            )

            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)

        train_loss = train_loss_sum / len(train_dataset)

        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for x, target_policy, target_value in val_loader:
                predicted_logits, predicted_value = model(x)

                log_probs = torch.log_softmax(predicted_logits, dim=1)
                policy_loss = -(
                    target_policy * log_probs
                ).sum(dim=1).mean()

                value_loss = value_loss_fn(
                    predicted_value, target_value
                )

                val_loss_sum += (
                    policy_loss.item() + value_loss.item()
                ) * x.size(0)

        val_loss = val_loss_sum / len(val_dataset)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"train loss: {train_loss:.4f} | "
                f"validation loss: {val_loss:.4f}"
            )

    torch.save(model.state_dict(), "tiny_colony_net.pth")
    print("\nSaved trained weights to tiny_colony_net.pth")


if __name__ == "__main__":
    main()
