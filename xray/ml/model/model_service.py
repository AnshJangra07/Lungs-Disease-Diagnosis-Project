import bentoml
import torch
from PIL import Image as PILImage
from torchvision import transforms

from xray.constant.training_pipeline import *

inference_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


@bentoml.service(name=BENTOML_SERVICE_NAME)
class XRayService:
    def __init__(self):
        self.model = bentoml.pytorch.load_model(
            BENTOML_MODEL_NAME,
            weights_only=False,
        )
        self.model.eval()

    @bentoml.api
    def predict(self, img: PILImage.Image) -> str:
        image = inference_transform(img.convert("RGB")).unsqueeze(0)

        with torch.no_grad():
            output = self.model(image)

        prediction_index = torch.argmax(output, dim=1).item()
        return PREDICTION_LABEL[prediction_index]
