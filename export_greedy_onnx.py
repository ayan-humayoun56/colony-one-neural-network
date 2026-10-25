
import torch
from tiny_colony_net import TinyColonyNet


def main():
    # Load the best checkpoint from GreedyAgent imitation training.
    model = TinyColonyNet()

    state_dict = torch.load(
        "tiny_colony_net_greedy.pth",
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    # Dummy input for export. The batch dimension remains dynamic.
    dummy_input = torch.zeros(
        2, 16, 8, 8, dtype=torch.float32
    )

    torch.onnx.export(
        model,
        dummy_input,
        "model_greedy.onnx",
        input_names=["observation"],
        output_names=["policy", "value"],
        dynamic_axes={
            "observation": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        external_data=False,
    )

    print("Exported model_greedy.onnx")


if __name__ == "__main__":
    main()

