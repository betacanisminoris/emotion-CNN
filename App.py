import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import v2

import json

# Model Path
PATH = "balanced_with_activation.pth"

st.title("Facial Emotion Recognition")


def predict(model, image):
    return model(image)


transformed_image = None


class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=(2, 2)),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=(2, 2)),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=(2, 2)),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 7),
        )

    def forward(self, x):
        return self.network(x)


def load_state(path):
    return torch.load(path, weights_only=True)


with open("mean_std.json", "r") as file:
    mean_std = json.load(file)

with open("idx_to_class.json", "r") as file:
    idx_to_class = json.load(file)


# Cache resource to avoid continuous loading of model
@st.cache_resource
def load_model(PATH):
    model = CNN(input_dim=3)
    model.load_state_dict(load_state(PATH))
    return model


model = load_model(PATH)
model.eval()

transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean_std["mean"], std=mean_std["std"]),
    ]
)

uploaded_file = st.file_uploader(
    "Upload an Image: ", max_upload_size=4, type=[".png", ".jpg"]
)
if uploaded_file is not None:
    if uploaded_file.type in ["image/png", "image/jpeg"]:
        image = Image.open(uploaded_file)

        transformed_image = transforms(image)
        st.image(image, width=300)

        # Display Result
        with torch.no_grad():
            prediction = torch.argmax(model(transformed_image.unsqueeze(0))).item()
            confidence = torch.max(
                torch.softmax(model(transformed_image.unsqueeze(0)), dim=-1)
            ).item()
            st.metric(
                "Predicted Emotion",
                f"{idx_to_class.get(str(prediction),"unknown emotion").title()} | Confidence: ({confidence:.2%})",
            )
