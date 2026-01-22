import torch
from src.toxic_comments.model import ToxicCommentsTransformer

CKPT_PATH = "models/best-checkpoint.ckpt"
ONNX_PATH = "models/model.onnx"

def main():
    """Convert the model to onnx"""
    print(f"Loading checkpoint from: {CKPT_PATH}")
    model = ToxicCommentsTransformer.load_from_checkpoint(CKPT_PATH)
    model.eval()

    dummy_input = (
        torch.randint(0, 1000, (1, 128)),
        torch.ones((1, 128), dtype=torch.long))

    print("Exporting to ONNX...")

    torch.onnx.export(
        model.model,
        dummy_input,
        ONNX_PATH,
        opset_version=18,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        }
    )
    
    print(f"Exported to {ONNX_PATH}")

if __name__ == "__main__":
    main()