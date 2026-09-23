import io

import bentoml
import torch
from bentoml.io import Image, Text
from PIL import Image as PILImage
from torchvision import transforms

from xray.constant.training_pipeline import *

bento_model = bentoml.pytorch.get(BENTOML_MODEL_NAME)

runner = bento_model.to_runner()

svc = bentoml.Service(name=BENTOML_SERVICE_NAME, runners=[runner])

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


@svc.api(input=Image(allowed_mime_types=["image/jpeg", "image/png"]), output=Text())
async def predict(img):
    b = io.BytesIO()

    img.save(b, "jpeg")

    im_bytes = b.getvalue()

    image = PILImage.open(io.BytesIO(im_bytes)).convert("RGB")
    image = inference_transform(image).unsqueeze(0)

    batch_ret = await runner.async_run(image)

    prediction_index = torch.argmax(batch_ret, dim=1).item()
    pred = PREDICTION_LABEL[prediction_index]

    return pred