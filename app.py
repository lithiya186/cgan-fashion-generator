import streamlit as st
import torch
import torch.nn as nn

# -------------------------------
# Generator (MATCHES TRAINING)
# -------------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_emb = nn.Embedding(10, 10)

        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.ReLU(),

            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        x = torch.cat([noise, label_input], dim=1)
        return self.model(x).view(-1, 1, 28, 28)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model = Generator()
    model.load_state_dict(torch.load("generator.pth", map_location="cpu"))
    model.eval()
    return model

generator = load_model()

# -------------------------------
# UI
# -------------------------------
st.title("Fashion Image Generator (cGAN)")
st.markdown("Generate AI fashion images in real-time")

classes = {
    0: "T-shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot"
}

option = st.selectbox("Select Category", list(classes.values()))
label = list(classes.keys())[list(classes.values()).index(option)]

num_images = st.slider("Number of images", 1, 10, 5)

# -------------------------------
# Generate Images
# -------------------------------
if st.button("Generate Images"):
    noise = torch.randn(num_images, 100)
    labels = torch.tensor([label] * num_images)

    with torch.no_grad():
        images = generator(noise, labels)

    cols = st.columns(5)

    for i in range(num_images):
        img = images[i][0].numpy()

        # Convert from [-1,1] to [0,1]
        img = (img + 1) / 2

        cols[i % 5].image(img, use_container_width=True)
