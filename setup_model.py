import torch
import torchvision.models as models
import urllib.request
import os

print("🚀 Starting Setup...")

# 1. Download & Save the Model (ResNet-18)
print("⬇️  Downloading ResNet-18 Model...")
try:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    dummy_input = torch.rand(1, 3, 224, 224)
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save("resnet18.pt")
    print("✅ Model saved: resnet18.pt")
except Exception as e:
    print(f"❌ Error saving model: {e}")

# 2. Download the Labels (Index to Name)
print("⬇️  Downloading Label Mappings...")
url = "https://raw.githubusercontent.com/pytorch/serve/master/examples/image_classifier/index_to_name.json"
try:
    urllib.request.urlretrieve(url, "index_to_name.json")
    if os.path.exists("index_to_name.json"):
        print("✅ Labels saved: index_to_name.json")
    else:
        print("❌ Error: Download appeared to finish but file is missing.")
except Exception as e:
    print(f"❌ Failed to download labels: {e}")

print("✨ Setup Complete! You are ready to package the model.")
