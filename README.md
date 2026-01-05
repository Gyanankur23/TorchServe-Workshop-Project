# 🐯 TorchServe Workshop: ResNet-18 Deployment

Welcome to the **Computer Vision Deployment Workshop**.

In this session, we will take a pre-trained PyTorch model (ResNet-18), package it for production using **TorchServe** (Docker), and build a user-friendly **Streamlit** frontend.

## 🛠️ Prerequisites
* **Docker Desktop** (Installed & Running)
* **Python 3.8+**
* **Git**

---

## 🎓 Workshop Steps

Follow these steps one by one.

### 🟢 STEP 1: Setup Environment
*Create a virtual environment and install dependencies.*

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
🟢 STEP 2: Download Model & Labels
Download the ResNet-18 model (.pt) and the ImageNet labels (.json).
python setup_model.py
✅ Check: Ensure resnet18.pt and index_to_name.json appear in your folder.
🟢 STEP 3: Package the Model
Convert the model files into a .mar (Model Archive) ready for serving.
torch-model-archiver --model-name resnet --version 1.0 --serialized-file resnet18.pt --handler image_classifier --extra-files index_to_name.json
✅ Check: Ensure resnet.mar appears in your folder.
🟢 STEP 4: Start the Server (Docker)
Launch the AI backend container.

1. First, clear any old servers:
docker stop $(docker ps -q)
2. Start the new server:
docker run --rm -d -p 8080:8080 -p 8081:8081 -v ${PWD}:/models pytorch/torchserve:latest torchserve --start --model-store /models --models resnet=resnet.mar --disable-token-auth
✅ Check: Run docker ps to verify the container is running.

🟢 STEP 5: Launch the App
Start the frontend interface.
streamlit run app.py
🎉 Activity: Go to Google Images, download a picture of a Panda, Plane, or Pizza, and drag it into the app!

🆘 Troubleshooting
1. "Bind for 0.0.0.0:8080 failed"

Fix: Run docker stop $(docker ps -q) to stop the old container.

2. "Module not found: torchvision"

Fix: Run pip install -r requirements.txt.

3. "503 Prediction Failed"

Fix: Re-run python setup_model.py and then re-run the torch-model-archiver command (Step 3).
