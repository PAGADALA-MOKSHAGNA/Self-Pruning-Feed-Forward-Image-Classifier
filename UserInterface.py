from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


DEFAULT_CHECKPOINT_DIR = Path("models")
DEFAULT_CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
DEFAULT_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
DEFAULT_CIFAR10_STD = (0.2470, 0.2435, 0.2616)
DEFAULT_IMAGE_SIZE = 32
DEFAULT_GATE_THRESHOLD = 0.10


class PrunableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        return F.linear(x, self.weight * gates, self.bias)


class PrunableCNN(nn.Module):
    def __init__(
        self,
        image_channels: int,
        conv_channel_1: int,
        conv_channel_2: int,
        classifier_input_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(image_channels, conv_channel_1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_channel_1, conv_channel_2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = PrunableLinear(classifier_input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


class PrunableMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int) -> None:
        super().__init__()
        self.hidden_layers = nn.ModuleList()

        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(PrunableLinear(previous_dim, hidden_dim))
            previous_dim = hidden_dim

        self.output_layer = PrunableLinear(previous_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)


def list_checkpoint_paths() -> list[Path]:
    if not DEFAULT_CHECKPOINT_DIR.exists():
        return []
    return sorted(DEFAULT_CHECKPOINT_DIR.glob("*.pt"))


def infer_model_name(checkpoint: dict[str, Any], checkpoint_path: Path) -> str:
    model_name = checkpoint.get("model_name")
    if isinstance(model_name, str) and model_name:
        return model_name

    file_name = checkpoint_path.name.lower()
    if "cnn" in file_name:
        return "PrunableCNN"
    if "mlp" in file_name:
        return "PrunableMLP"
    raise ValueError("Could not infer model architecture from the checkpoint.")


def safe_torch_load(checkpoint_path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint is not a dictionary with model metadata.")
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain 'model_state_dict'.")
    return checkpoint


def build_model_from_checkpoint(
    checkpoint: dict[str, Any], checkpoint_path: Path
) -> tuple[nn.Module, dict[str, Any]]:
    model_name = infer_model_name(checkpoint, checkpoint_path)
    class_names = checkpoint.get("class_names", DEFAULT_CIFAR10_CLASSES)
    num_classes = int(checkpoint.get("num_classes", len(class_names)))

    if model_name == "PrunableCNN":
        image_channels = int(checkpoint.get("image_channels", 3))
        conv_channel_1 = int(checkpoint.get("conv_channel_1", 32))
        conv_channel_2 = int(checkpoint.get("conv_channel_2", 64))
        classifier_input_dim = int(checkpoint.get("classifier_input_dim", 64 * 16 * 16))
        model = PrunableCNN(
            image_channels=image_channels,
            conv_channel_1=conv_channel_1,
            conv_channel_2=conv_channel_2,
            classifier_input_dim=classifier_input_dim,
            num_classes=num_classes,
        )
    elif model_name == "PrunableMLP":
        input_dim = int(checkpoint.get("input_dim", 3 * 32 * 32))
        hidden_dims = [
            int(checkpoint[key])
            for key in ("hidden_dim_0", "hidden_dim_1", "hidden_dim_2")
            if key in checkpoint
        ]
        if not hidden_dims:
            hidden_dims = [512, 256]
        model = PrunableMLP(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metadata = {
        "checkpoint_path": checkpoint_path,
        "model_name": model_name,
        "class_names": class_names,
        "num_classes": num_classes,
        "image_size": int(checkpoint.get("image_size", DEFAULT_IMAGE_SIZE)),
        "cifar10_mean": tuple(checkpoint.get("cifar10_mean", DEFAULT_CIFAR10_MEAN)),
        "cifar10_std": tuple(checkpoint.get("cifar10_std", DEFAULT_CIFAR10_STD)),
        "gate_threshold": float(checkpoint.get("gate_threshold", DEFAULT_GATE_THRESHOLD)),
        "best_result": checkpoint.get("best_result", {}),
        "use_data_augmentation": bool(checkpoint.get("use_data_augmentation", False)),
    }
    return model, metadata


@st.cache_resource(show_spinner=False)
def load_model_bundle(checkpoint_path_str: str) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path_str)
    checkpoint = safe_torch_load(checkpoint_path)
    return build_model_from_checkpoint(checkpoint, checkpoint_path)


def collect_gate_values(model: nn.Module) -> torch.Tensor:
    gate_values = []
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gate_values.append(torch.sigmoid(module.gate_scores.detach().cpu()).flatten())

    if not gate_values:
        return torch.empty(0)
    return torch.cat(gate_values)


def compute_gate_metrics(gate_values: torch.Tensor, threshold: float) -> dict[str, float]:
    if gate_values.numel() == 0:
        return {
            "sparsity_pct": 0.0,
            "active_parameters": 0,
            "total_prunable_parameters": 0,
            "mean_gate": 0.0,
            "median_gate": 0.0,
            "min_gate": 0.0,
            "max_gate": 0.0,
        }

    total_gates = gate_values.numel()
    pruned_gates = int((gate_values < threshold).sum().item())
    active_gates = total_gates - pruned_gates
    return {
        "sparsity_pct": 100.0 * pruned_gates / total_gates,
        "active_parameters": active_gates,
        "total_prunable_parameters": total_gates,
        "mean_gate": float(gate_values.mean().item()),
        "median_gate": float(gate_values.median().item()),
        "min_gate": float(gate_values.min().item()),
        "max_gate": float(gate_values.max().item()),
    }


def build_preprocess_transform(image_size: int, mean: tuple[float, ...], std: tuple[float, ...]):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def preprocess_image(image: Image.Image, metadata: dict[str, Any]) -> torch.Tensor:
    image = image.convert("RGB")
    transform = build_preprocess_transform(
        image_size=int(metadata["image_size"]),
        mean=tuple(metadata["cifar10_mean"]),
        std=tuple(metadata["cifar10_std"]),
    )
    return transform(image).unsqueeze(0)


def summarize_predictions(
    probabilities: torch.Tensor, class_names: list[str], top_k: int
) -> tuple[list[dict[str, Any]], float, float]:
    top_k = min(top_k, probabilities.numel())
    top_probs, top_indices = torch.topk(probabilities, k=top_k)

    rows = []
    for rank, (index, confidence) in enumerate(zip(top_indices.tolist(), top_probs.tolist()), start=1):
        rows.append(
            {
                "Rank": rank,
                "Class": class_names[index],
                "Class index": index,
                "Confidence": f"{confidence:.2%}",
            }
        )

    entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum().item())
    margin = float(top_probs[0].item() - top_probs[1].item()) if top_k > 1 else float(top_probs[0].item())
    return rows, entropy, margin


def plot_probability_chart(rows: list[dict[str, Any]]) -> None:
    labels = [row["Class"] for row in rows]
    values = [float(row["Confidence"].rstrip("%")) / 100.0 for row in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color="#2563eb")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Confidence")
    ax.set_title("Top class probabilities")
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.1%}",
            ha="center",
            va="bottom",
        )

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_gate_histogram(gate_values: torch.Tensor, threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(gate_values.numpy(), bins=40, color="#7c3aed", edgecolor="white")
    ax.axvline(
        threshold,
        color="#dc2626",
        linestyle="--",
        linewidth=2,
        label=f"threshold = {threshold:.2f}",
    )
    ax.set_title("Gate value distribution")
    ax.set_xlabel("Gate value")
    ax.set_ylabel("Number of weights")
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_sidebar(metadata: dict[str, Any], gate_values: torch.Tensor) -> tuple[float, dict[str, float]]:
    with st.sidebar:
        st.header("Model")
        st.write(f"**Architecture:** {metadata['model_name']}")
        st.write(f"**Checkpoint:** `{metadata['checkpoint_path'].name}`")
        st.write(f"**Classes:** {metadata['num_classes']}")
        st.write(f"**Augmentation in training:** {metadata['use_data_augmentation']}")

        gate_threshold = st.slider(
            "Gate threshold",
            min_value=0.01,
            max_value=0.50,
            value=float(metadata["gate_threshold"]),
            step=0.01,
            help="Used to estimate functional sparsity from the learned sigmoid gates.",
        )

        gate_metrics = compute_gate_metrics(gate_values, gate_threshold)

        st.subheader("Pruning Metrics")
        st.metric("Sparsity", f"{gate_metrics['sparsity_pct']:.2f}%")
        st.metric("Mean gate", f"{gate_metrics['mean_gate']:.3f}")
        st.metric("Median gate", f"{gate_metrics['median_gate']:.3f}")
        st.metric("Prunable params", f"{gate_metrics['total_prunable_parameters']:,}")

        with st.expander("Checkpoint details", expanded=False):
            best_result = metadata.get("best_result", {})
            if best_result:
                st.write(best_result)
            else:
                st.caption("No extra checkpoint metadata was stored.")

    return gate_threshold, gate_metrics


def main() -> None:
    st.set_page_config(
        page_title="Self-Pruning Classifier",
        page_icon="🧠",
        layout="wide",
    )

    st.title("Self-Pruning CIFAR-10 Classifier")
    st.caption(
        "Upload an image to run inference with the saved checkpoint and inspect prediction confidence "
        "together with self-pruning metrics."
    )

    checkpoint_paths = list_checkpoint_paths()
    if not checkpoint_paths:
        st.error("No model checkpoints were found in the `models/` directory.")
        st.stop()

    checkpoint_names = [path.name for path in checkpoint_paths]
    default_index = next((index for index, path in enumerate(checkpoint_paths) if "cnn" in path.name.lower()), 0)
    selected_checkpoint_name = st.selectbox("Choose a checkpoint", checkpoint_names, index=default_index)
    selected_checkpoint_path = next(path for path in checkpoint_paths if path.name == selected_checkpoint_name)

    top_k = st.slider("How many top predictions should be displayed?", min_value=3, max_value=5, value=3)

    try:
        model, metadata = load_model_bundle(str(selected_checkpoint_path.resolve()))
    except Exception as error:
        st.error(f"Failed to load the selected checkpoint: {error}")
        st.stop()

    gate_values = collect_gate_values(model)
    gate_threshold, gate_metrics = render_sidebar(metadata, gate_values)

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        help="The image will be resized to the model input size before inference.",
    )

    if uploaded_file is None:
        st.info("Upload an image to see the predicted class, confidence score, and pruning metrics.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = preprocess_image(image, metadata)

    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    class_names = list(metadata["class_names"])
    prediction_rows, entropy, margin = summarize_predictions(probabilities, class_names, top_k=top_k)
    top_prediction = prediction_rows[0]
    predicted_index = int(top_prediction["Class index"])
    confidence_value = probabilities[predicted_index].item()

    preview_column, summary_column = st.columns([1, 1.25])

    with preview_column:
        st.image(image, caption=f"Uploaded image ({image.width} x {image.height})", use_container_width=True)
        st.caption(
            f"Preprocessed as RGB, resized to {metadata['image_size']} x {metadata['image_size']}, "
            "and normalized with CIFAR-10 statistics."
        )

    with summary_column:
        st.subheader("Prediction")
        metric_columns = st.columns(4)
        metric_columns[0].metric("Predicted class", top_prediction["Class"])
        metric_columns[1].metric("Confidence", f"{confidence_value:.2%}")
        metric_columns[2].metric("Top-2 margin", f"{margin:.2%}")
        metric_columns[3].metric("Entropy", f"{entropy:.3f}")

        pruning_columns = st.columns(4)
        pruning_columns[0].metric("Model sparsity", f"{gate_metrics['sparsity_pct']:.2f}%")
        pruning_columns[1].metric("Mean gate", f"{gate_metrics['mean_gate']:.3f}")
        pruning_columns[2].metric("Active gates", f"{gate_metrics['active_parameters']:,}")
        pruning_columns[3].metric("Total prunable", f"{gate_metrics['total_prunable_parameters']:,}")

        st.subheader("Top predictions")
        st.table(prediction_rows)
        plot_probability_chart(prediction_rows)

    st.subheader("Model pruning diagnostics")
    diagnostics_columns = st.columns(3)
    diagnostics_columns[0].metric("Min gate", f"{gate_metrics['min_gate']:.3f}")
    diagnostics_columns[1].metric("Median gate", f"{gate_metrics['median_gate']:.3f}")
    diagnostics_columns[2].metric("Max gate", f"{gate_metrics['max_gate']:.3f}")

    if gate_values.numel() > 0:
        plot_gate_histogram(gate_values, gate_threshold)


if __name__ == "__main__":
    main()