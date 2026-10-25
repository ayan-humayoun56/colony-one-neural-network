import torch
from tiny_colony_net import TinyColonyNet


def main():
    model = TinyColonyNet()
    state_dict = torch.load(
        "tiny_colony_net.pth",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    # Use two examples while tracing, but make the batch axis dynamic.
    dummy_input = torch.zeros(2, 16, 8, 8, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        "model.onnx",
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

    print("Exported model.onnx")


if __name__ == "__main__":
    main()
