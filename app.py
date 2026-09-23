# fast api application for x-ray image classification

import torch
from xray.ml.model.arch import Net  # Ensure this path is correct
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize and load model weights
model = Net().to(device)
model_path = Path(__file__).resolve().with_name("xray_model.pth")
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Prediction label mapping
label_map = {
    0: "Normal",
    1: "Pneumonia"
}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
    except (OSError, Image.UnidentifiedImageError) as error:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image") from error

    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        prediction_index = torch.argmax(output, dim=1).item()
        prediction_label = label_map.get(prediction_index, "Unknown")
    
    return {
        "prediction_index": prediction_index,
        "prediction_label": prediction_label
    }
