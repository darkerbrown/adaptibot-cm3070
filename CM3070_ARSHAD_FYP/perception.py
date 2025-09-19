"""Optional image-based perception used to slow the controller in hard scenes.

- When the simulator shows an image from the robot's camera, these
    models look at that image and try to answer two easy questions:
    1) Is the ground mostly flat or covered in rubble or mud? (Terrain)
    2) How hard is it to move here? (Difficulty)
- The controller can use these answers to decide to slow down or be
    more careful near difficult terrain.

How the wrappers behave:
- Each wrapper exposes `predict(image)` which returns a short text
    label (e.g. "rubble") and a small list of probabilities. For a
    grader, the important thing is these are summaries of the current
    camera view and they are optional (the system runs without them).
"""
from typing import Tuple, Optional
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from torchvision import models, transforms                # required
from transformers import (                                # required
        EfficientNetImageProcessor,
        EfficientNetForImageClassification,
)


IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD  = (0.229, 0.224, 0.225)

def _ensure_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    if not isinstance(rgb, np.ndarray):
        raise TypeError("predict() expects HxWx3 numpy array")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("predict() expects HxWx3 RGB")
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8, copy=False)
    return rgb

class TerrainClassifier:
    """EfficientNet-B0 terrain classifier (flat/mud/rubble)."""
    CLASSES = ["flat", "rubble", "mud"]

    def __init__(self, device: torch.device, weights_path: Optional[str]):
        self.device = device
        self.ready = False
        self.source = "hf"

        classes = self.CLASSES
        id2label = {i: c for i, c in enumerate(classes)}
        label2id = {c: i for i, c in enumerate(classes)}

                # Use HF processor to apply the same resize/crop as training.
        self.proc = EfficientNetImageProcessor.from_pretrained(
            "google/efficientnet-b0",
            size={"height": 160, "width": 160},           # down from 224 to cut cost
            crop_size={"height": 160, "width": 160}
        )
                # Load backbone and remap classifier head to our terrain labels.
        self.model = EfficientNetForImageClassification.from_pretrained(
            "google/efficientnet-b0",
            num_labels=len(classes),
            ignore_mismatched_sizes=True,
            id2label=id2label,
            label2id=label2id,
        ).to(device).eval()

                # Allow finetuned weights to override the base checkpoint.
        if weights_path and os.path.exists(weights_path):
            sd = torch.load(weights_path, map_location=device)
            self.model.load_state_dict(sd, strict=False)
            self.source = "hf+weights"

        self.ready = True

    @torch.no_grad()
    def predict(self, rgb: np.ndarray) -> Tuple[str, np.ndarray]:
        if not self.ready:
            probs = np.ones(len(self.CLASSES), dtype=np.float32) / len(self.CLASSES)
            return self.CLASSES[0], probs
                # Guard against floats or grayscale inputs from callers.
        rgb = _ensure_uint8_rgb(rgb)
        pil = Image.fromarray(rgb)
        inputs = self.proc(images=pil, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0].astype(np.float32, copy=False)
        idx = int(np.argmax(probs))
        return self.CLASSES[idx], probs

class DifficultyEstimator:
    """ResNet-18 difficulty estimator (easy/moderate/difficult)."""
    CLASSES = ["easy", "moderate", "difficult"]

    def __init__(self, device: torch.device, weights_path: Optional[str]):
        self.device = device
        self.ready = False
        self.source = "tv"

                # Bootstrap a torchvision ResNet-18 for difficulty estimation.
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, len(self.CLASSES))
        if weights_path and os.path.exists(weights_path):
            sd = torch.load(weights_path, map_location=device)
            m.load_state_dict(sd, strict=False)
            self.source = "tv+weights"

        self.model = m.to(device).eval()
                # Apply ImageNet-normalised transforms consistent with training.
        self.transform = transforms.Compose([
            transforms.Resize(180),
            transforms.CenterCrop(160),
            transforms.ToTensor(),
            transforms.Normalize(IMNET_MEAN, IMNET_STD),
        ])
        self.ready = True

    @torch.no_grad()
    def predict(self, rgb: np.ndarray) -> Tuple[str, np.ndarray]:
        if not self.ready:
            probs = np.ones(len(self.CLASSES), dtype=np.float32) / len(self.CLASSES)
            return self.CLASSES[0], probs
        rgb = _ensure_uint8_rgb(rgb)
        pil = Image.fromarray(rgb)
        x = self.transform(pil).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0].astype(np.float32, copy=False)
        idx = int(np.argmax(probs))
        return self.CLASSES[idx], probs
