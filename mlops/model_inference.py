import cv2
import numpy as np
import torch

from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from torchvision.transforms import functional as TF


class RoadSegmenter:

    def __init__(
        self,
        checkpoint_path,
        device=None,
        image_size=512,
        threshold=0.5
    ):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
        )

        self.image_size = image_size
        self.threshold = threshold

        self.model = deeplabv3_resnet50(
            weights=None,
            num_classes=1
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def preprocess(self, image):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_size = image.shape[:2]

        resized = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR
        )

        tensor = TF.to_tensor(resized)

        tensor = TF.normalize(
            tensor,
            self.mean,
            self.std
        )

        return tensor.unsqueeze(0), original_size

    @torch.no_grad()
    def predict(self, image):

        tensor, original_size = self.preprocess(image)

        tensor = tensor.to(self.device)

        output = self.model(tensor)["out"]

        mask = torch.sigmoid(output)

        mask = mask.squeeze().cpu().numpy()

        mask = (mask > self.threshold).astype(np.uint8)

        mask = cv2.resize(
            mask,
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST
        )

        return mask

    def overlay(self, image, mask):

        overlay = image.copy()

        overlay[mask == 1] = (0, 255, 0)

        return cv2.addWeighted(
            image,
            0.7,
            overlay,
            0.3,
            0
        )