from PIL import Image
import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

print(torch.__version__, device)
classes = ["Bio-degredable", "Non Bio-degredable"]

# Build neural network
class EnvironmentClassifier(nn.Module):
  def __init__(self, reps : int = 1, image_dims = []):
    super(EnvironmentClassifier, self).__init__()
    self.flatten = nn.Flatten()

    self.Classifier = nn.ModuleList()
    self.MakeModel(reps, image_dims)


  def MakeModel(self, Reps, Image_dims):
    In_Channel = 3
    Out_Channel = 16
    H, W = Image_dims[1], Image_dims[2]  # local copy

    for i in range(Reps):
        self.Classifier.append(nn.Sequential(
            nn.Conv2d(in_channels=In_Channel, out_channels=Out_Channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ))

        In_Channel = Out_Channel
        Out_Channel *= 2
        H //= 2
        W //= 2

    self.Classifier.append(nn.Sequential(
        nn.Flatten(),
        nn.Linear(In_Channel * H * W, 512),
        nn.ReLU(),
        nn.Linear(512, 1)
    ))

  def forward(self, x):
    for layer in self.Classifier:
      x = layer(x)
    return x


def Modelload(PthPath : str, Model : nn.Module):
    state_dict = torch.load(PthPath, weights_only=True)
    Model.load_state_dict(state_dict)
    
# create argument parser
parser = argparse.ArgumentParser(description="Run model inference on an image.")

parser.add_argument(
    "model_path",  # positional argument
    type=str,
    help="Path to the .pth model file"
)

parser.add_argument(
    "image_path",  # positional argument
    type=str,
    help="Path to the input image file"
)

args = parser.parse_args()

print(f"Model path: {args.model_path}")
print(f"Image path: {args.image_path}")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Predicts the image u give
def pred(ImagePath):
  # Open an image file
  img = Image.open(ImagePath)
  img_tensor = transform(img)
  img_tensor.unsqueeze_(dim=0)

  logits = Model(img_tensor.to(device))
  pred = (torch.sigmoid(logits) >= 0.5).float()

  label = classes[int(pred.item())]

  # Display the image
  
  # add label alongside (above or below or side)
  plt.title(f"Pred: {label}", fontsize=14)

  # Display image inline
  plt.imshow(img)
  plt.axis('off')  # optional: hide axes
  plt.show()
  
if __name__ == "__main__":
  Model = EnvironmentClassifier(reps=2, image_dims=[3, 64, 64]).to(device)
  print(Model)
  print(torch.cuda.is_available())
  print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
  
  Modelload(PthPath=args.model_path, Model=Model)

  pred(ImagePath=args.image_path)