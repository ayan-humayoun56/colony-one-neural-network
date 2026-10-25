import copy
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

    # Load the GreedyAgent dataset.
    data = np.load("greedy_data.npz")

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

    # Split into training and validation sets.
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False
    )

    model = TinyColonyNet().to(device)

    # A smaller learning rate makes parameter updates more gradual.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0003
    )

    value_loss_fn = nn.MSELoss()

    max_epochs = 60
    patience = 8
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(max_epochs):
        # -------------------------
        # Training
        # -------------------------
        model.train()
        train_loss_sum = 0.0

        for x, target_policy, target_value in train_loader:
            x = x.to(device)
            target_policy = target_policy.to(device)
            target_value = target_value.to(device)

            predicted_logits, predicted_value = model(x)

            log_probs = torch.log_softmax(
                predicted_logits, dim=1
            )

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

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for x, target_policy, target_value in val_loader:
                x = x.to(device)
                target_policy = target_policy.to(device)
                target_value = target_value.to(device)

                predicted_logits, predicted_value = model(x)

                log_probs = torch.log_softmax(
                    predicted_logits, dim=1
                )

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

        # Save a copy whenever validation improves.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:02d}/{max_epochs} | "
                f"train loss: {train_loss:.4f} | "
                f"validation loss: {val_loss:.4f} | "
                f"best epoch: {best_epoch}"
            )

        # Stop if validation has not improved for 8 epochs.
        if epochs_without_improvement >= patience:
            print(
                f"\nEarly stopping at epoch {epoch + 1}. "
                f"Best epoch was {best_epoch}."
            )
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    # Restore and save the best validation checkpoint.
    model.load_state_dict(best_state)
    torch.save(
        model.state_dict(),
        "tiny_colony_net_greedy.pth"
    )

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Best epoch: {best_epoch}")
    print("Saved best weights to tiny_colony_net_greedy.pth")


if __name__ == "__main__":
    main()