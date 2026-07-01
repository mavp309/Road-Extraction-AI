import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp


class RoadSegmenter:

    def __init__(
        self,
        checkpoint_path,
        device=None,
        image_size=512,
        threshold=0.5,
        encoder="resnet34"
    ):

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.image_size = image_size
        self.threshold = threshold

        # EXACTLY the same model used during training
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=None,      # IMPORTANT
            in_channels=3,
            classes=1,
        )

        state_dict = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        self.mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32,
        )

        self.std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )

    def preprocess(self, image):

        original_size = image.shape[:2]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        image = image.astype(np.float32) / 255.0

        image = (image - self.mean) / self.std

        image = np.transpose(image, (2, 0, 1))

        tensor = torch.from_numpy(image).float()

        tensor = tensor.unsqueeze(0)

        return tensor, original_size

    @torch.no_grad()
    def predict(self, image, as_probability: bool = False):

        tensor, original_size = self.preprocess(image)

        tensor = tensor.to(self.device)

        logits = self.model(tensor)

        probs = torch.sigmoid(logits)

        prob = probs.squeeze().cpu().numpy()

        if as_probability:
            prob_resized = cv2.resize(
                prob,
                (original_size[1], original_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            return prob_resized.astype(np.float32)

        mask = (prob > self.threshold).astype(np.uint8)

        mask = cv2.resize(
            mask,
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        return mask

    def overlay(self, image, mask):
        # Accept either a binary mask (0/1 or 0/255 uint8) or a
        # probability map in [0,1]. Blend a green overlay according
        # to per-pixel confidence.
        img_f = image.astype(np.float32)

        # Normalize mask to 0..1 float alpha
        if mask.dtype == np.uint8:
            alpha = mask.astype(np.float32) / 255.0 if mask.max() > 1 else mask.astype(np.float32)
        else:
            alpha = mask.astype(np.float32)

        if alpha.ndim == 2:
            alpha = np.expand_dims(alpha, axis=2)

        color = np.array([0.0, 255.0, 0.0], dtype=np.float32)

        tint_strength = 0.6

        blended = img_f * (1.0 - tint_strength * alpha) + color * (tint_strength * alpha)

        return blended.astype(np.uint8)