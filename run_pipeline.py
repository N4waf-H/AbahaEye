\
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from tqdm import tqdm

from openai_severity import analyze_image

LABELS_FILE = "outputs/label_map.json"

def load_classifier(model_path: str, num_classes: int = 2, device: str = "cpu"):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model

def predict_accident(model, image_path: str, device: str = "cpu", img_size: int = 224) -> Dict[str, Any]:
    tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    x = tfms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return {"probs": probs, "pred_idx": int(probs.argmax())}

def rule_based_dispatch_fallback(severity_result: Dict[str, Any]) -> List[str]:
    # If the model returns empty or seems off, apply conservative rules
    sev = severity_result.get("severity_level", "none")
    hazards = severity_result.get("hazards", [])
    injuries = bool(severity_result.get("injuries_likely", False))

    dispatch = set(severity_result.get("recommended_dispatch", []))

    if sev in ["moderate", "severe", "catastrophic"]:
        dispatch.add("police")
    if injuries or sev in ["severe", "catastrophic"]:
        dispatch.add("ambulance")
    if "fire_or_smoke" in hazards or "vehicle_on_fire" in hazards:
        dispatch.add("fire_department")
    if "fallen_pole_or_power_lines" in hazards:
        dispatch.add("utility_crew")
    if "blocked_lanes" in hazards:
        dispatch.add("traffic_management")
    if sev == "minor" and not injuries:
        dispatch.add("accident_rating_agent")

    return sorted(dispatch)

def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # Load label map (idx->class)
    with open(args.labels, "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)

    num_classes = len(idx_to_class)
    model = load_classifier(args.model_path, num_classes=num_classes, device=device)

    rows = []
    image_paths = sorted([str(p) for p in Path(args.images_dir).glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    for img_path in tqdm(image_paths, desc="Processing images"):
        clf = predict_accident(model, img_path, device=device)
        pred_idx = clf["pred_idx"]
        probs = clf["probs"]
        pred_label = idx_to_class[str(pred_idx)] if isinstance(idx_to_class, dict) else idx_to_class[pred_idx]
        accident_prob = float(probs[pred_idx])

        severity = {}
        dispatch = []
        if pred_label.lower() == "accident" and accident_prob >= args.threshold:
            severity = analyze_image(img_path, system_prompt_path=args.system_prompt)
            # Enforce fallback/augmentation rules
            dispatch = rule_based_dispatch_fallback(severity)
        else:
            severity = {
                "severity_level": "none",
                "injuries_likely": False,
                "hazards": [],
                "recommended_dispatch": [],
                "confidence": 1.0 - float(accident_prob),
                "reasoning_short": "Classifier indicates no accident."
            }
            dispatch = []

        rows.append({
            "image": os.path.basename(img_path),
            "accident_pred": pred_label,
            "accident_prob": round(accident_prob, 4),
            "severity_level": severity.get("severity_level"),
            "injuries_likely": severity.get("injuries_likely"),
            "hazards": ",".join(severity.get("hazards", [])),
            "dispatch_units": ",".join(dispatch),
            "confidence": severity.get("confidence"),
            "reasoning_short": severity.get("reasoning_short"),
        })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, "abhaeye_results.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, required=True, help="Folder with input images")
    ap.add_argument("--model_path", type=str, default="outputs/accident_classifier.pt")
    ap.add_argument("--labels", type=str, default=LABELS_FILE)
    ap.add_argument("--out_dir", type=str, default="./outputs")
    ap.add_argument("--threshold", type=float, default=0.6, help="Accident prob threshold to trigger severity step")
    ap.add_argument("--system_prompt", type=str, default="./prompts/severity_prompt.md")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    main(args)
