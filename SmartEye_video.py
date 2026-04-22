# =============================================================================
# SmartEye – Aircraft Damage Detection & Repair Report Generator
# Usage: python smarteye_video.py --video <path_to_video.mp4>
#        python smarteye_video.py --video video.47.14.mp4
# =============================================================================

# ── pip installs (run once in your terminal before executing this script) ─────
# pip install ultralytics langchain-groq langchain-core albumentations
#             torch torchvision joblib opencv-python matplotlib pydantic
# =============================================================================

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from typing import List, Literal

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models
from ultralytics import YOLO

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


# ── CONFIG ────────────────────────────────────────────────────────────────────

GROQ_API_KEY   = "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"
DENT_MODEL     = "best_recall.pt"
SCRATCH_MODEL  = "(Model_1_1)_Scratch_Detection_Model.pt"
DEPTH_MODEL    = "best_model.pth"
DEPTH_SCALER   = "depth_scaler.pkl"

# Aircraft / session metadata (edit as needed or pass via CLI)
DEFAULT_META = {
    "session_id":   f"SE-{datetime.now(timezone.utc).strftime('%Y-%m%d')}-{str(uuid.uuid4())[:4].upper()}",
    "report_date":  datetime.now(timezone.utc).isoformat(),
    "location":     "UNKNOWN",
    "aircraft": {
        "registration": "UNKNOWN",
        "type":         "UNKNOWN",
        "airline":      "UNKNOWN",
    },
}

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── MODELS ────────────────────────────────────────────────────────────────────

try:
    print(f"Loading YOLO dent model: {DENT_MODEL}")
    dent_yolo = YOLO(DENT_MODEL).to(device)
    print("✓ Dent model loaded")
except FileNotFoundError as e:
    print(f"❌ ERROR: Dent model file not found: {DENT_MODEL}")
    print(f"   Make sure the file exists at: {os.path.abspath(DENT_MODEL)}")
    raise
except Exception as e:
    print(f"❌ ERROR loading dent model: {e}")
    raise

try:
    print(f"Loading YOLO scratch model: {SCRATCH_MODEL}")
    scratch_yolo = YOLO(SCRATCH_MODEL).to(device)
    print("✓ Scratch model loaded")
except FileNotFoundError as e:
    print(f"❌ ERROR: Scratch model file not found: {SCRATCH_MODEL}")
    print(f"   Make sure the file exists at: {os.path.abspath(SCRATCH_MODEL)}")
    raise
except Exception as e:
    print(f"❌ ERROR loading scratch model: {e}")
    raise

try:
    print(f"Loading depth scaler: {DEPTH_SCALER}")
    scaler = joblib.load(DEPTH_SCALER)
    print("✓ Depth scaler loaded")
except FileNotFoundError as e:
    print(f"❌ ERROR: Depth scaler file not found: {DEPTH_SCALER}")
    print(f"   Make sure the file exists at: {os.path.abspath(DEPTH_SCALER)}")
    raise
except Exception as e:
    print(f"❌ ERROR loading depth scaler: {e}")
    raise


class DentRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights="IMAGENET1K_V1")
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.layer4.parameters():
            p.requires_grad = True
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(), nn.ReLU(),
            nn.Linear(128, 32),          nn.ReLU(), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.regressor(self.backbone(x))


try:
    print(f"Loading depth model: {DEPTH_MODEL}")
    depth_model = DentRegressor().to(device)
    depth_model.load_state_dict(torch.load(DEPTH_MODEL, map_location=device))
    depth_model.eval()
    print("✓ Depth model loaded")
except FileNotFoundError as e:
    print(f"❌ ERROR: Depth model file not found: {DEPTH_MODEL}")
    print(f"   Make sure the file exists at: {os.path.abspath(DEPTH_MODEL)}")
    raise
except Exception as e:
    print(f"❌ ERROR loading depth model: {e}")
    raise

depth_tfms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


@torch.no_grad()
def predict_depth(crop: np.ndarray) -> float:
    t = depth_tfms(image=crop)["image"].unsqueeze(0).to(device)
    pred = depth_model(t).cpu().numpy()
    return float(scaler.inverse_transform(pred)[0][0])


# ── PYDANTIC SCHEMAS ──────────────────────────────────────────────────────────

class DamageRecommendation(BaseModel):
    damage_id:              str
    geometry_summary:       str
    risk_assessment:        str
    recommended_repair_type: str
    justification:          str
    escalation_required:    bool
    confidence_level:       Literal["High", "Medium", "Low"]


class OverallRepairStrategy(BaseModel):
    general_repair_approach:    str
    interaction_between_damages: str
    inspection_requirements:    str
    structural_risk_level:      Literal["Low", "Moderate", "High"]
    engineering_review_required: bool


class PrioritizationItem(BaseModel):
    damage_id:    str
    priority_rank: int
    justification: str


class RepairRecommendationOutput(BaseModel):
    damages:                List[DamageRecommendation]
    overall_repair_strategy: OverallRepairStrategy
    prioritization:         List[PrioritizationItem]


# ── LLM SETUP ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a certified aircraft structural repair engineer assistant.

==============================
ROLE
==============================
You analyze aircraft damage geometries and generate conservative repair
recommendations strictly based on the provided data.

You are NOT allowed to:
- Assume aircraft model
- Assume structural load paths
- Assume material thickness
- Assume internal substructure
- Invent missing measurements

Unless explicitly stated, assume:
- Aluminum fuselage skin
- Non-critical structural zone
- No prior repairs

If information is insufficient for safe classification,
you MUST recommend engineering inspection.

==============================
ANALYSIS CONSTRAINTS
==============================
You must:
- Evaluate each damage independently.
- Base conclusions ONLY on geometric properties:
  size, aspect ratio, depth, proximity, overlap, clustering.
- Identify structural concerns such as:
  - Crack initiation risk
  - Edge distance concerns
  - Interaction between damages
  - Surface distortion risk
- If geometry approaches typical repair thresholds,
  default to conservative repair classification.
- If ambiguity exists, escalate.

Never fabricate dimensions or values not present in the report.

==============================
DECISION GUIDELINES (Conservative Aerospace Logic)
==============================
- Small superficial shallow damage → blending or minor surface repair
- Deeper localized dent → cold working or local repair
- Large area deformation → doubler or patch consideration
- Clustered or interacting damage → engineering review required
- Any uncertainty → escalate

==============================
OUTPUT REQUIREMENTS
==============================
Your output MUST strictly follow the structured schema provided.
Each field must be filled using only geometry-based reasoning.

Confidence Level:
- High   → geometry clearly supports decision
- Medium → some geometric ambiguity
- Low    → geometry insufficient, escalation required
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Aircraft Damage Report:\n{damage_report}\n\n"
               "Generate a fully structured repair recommendation. "
               "Do not add explanations outside the required structured format."),
])

try:
    print("Initializing LLM (ChatGroq)...")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is empty!")
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    print("✓ LLM initialized")
except Exception as e:
    print(f"❌ ERROR initializing LLM: {e}")
    raise


def get_repair_recommendation(damage_list: list) -> RepairRecommendationOutput:
    try:
        print(f"Sending {len(damage_list)} damages to LLM for analysis...")
        chain = prompt | llm.with_structured_output(RepairRecommendationOutput)
        result = chain.invoke({"damage_report": damage_list})
        print("✓ LLM response received")
        return result
    except Exception as e:
        print(f"❌ ERROR calling LLM: {e}")
        print(f"   Damage report sent: {damage_list}")
        raise


# ── VIDEO PROCESSING ──────────────────────────────────────────────────────────

def get_best_frame(video_path: str, conf: float = 0.35, frame_skip: int = 5):
    cap = cv2.VideoCapture(video_path)
    max_score, best_frame, frame_idx = 0, None, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dent_res    = dent_yolo.predict(rgb,    conf=conf, verbose=False)[0]
        scratch_res = scratch_yolo.predict(rgb, conf=conf, verbose=False)[0]
        score = 0

        if dent_res.boxes is not None:
            for box, c in zip(dent_res.boxes.xyxy, dent_res.boxes.conf):
                x1, y1, x2, y2 = box
                score += float(c) * (x2 - x1) * (y2 - y1) * 2

        if scratch_res.boxes is not None:
            for box, c in zip(scratch_res.boxes.xyxy, scratch_res.boxes.conf):
                x1, y1, x2, y2 = box
                score += float(c) * (x2 - x1) * (y2 - y1)

        if score > max_score:
            max_score = score
            best_frame = frame.copy()

        frame_idx += 1

    cap.release()
    print(f"Best score: {max_score}")
    return best_frame


def detect_damage_frame(frame) -> list:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dent_res    = dent_yolo.predict(rgb,    conf=0.25, verbose=False)[0]
    scratch_res = scratch_yolo.predict(rgb, conf=0.25, verbose=False)[0]

    damage_list, did = [], 0

    if scratch_res.boxes is not None:
        for box in scratch_res.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            if w < 20 or h < 20:
                continue
            damage_list.append({
                "damage_id":          f"D{did}",
                "damage_type":        "scratch",
                "width_px":           w,
                "height_px":          h,
                "area_px":            w * h,
                "aspect_ratio":       round(w / max(h, 1), 2),
                "confidence":         1.0,
                "estimated_depth_mm": 0,
            })
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            did += 1

    if dent_res.boxes is not None:
        for box in dent_res.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            if w < 20 or h < 20:
                continue
            crop = rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            depth = predict_depth(crop)
            damage_list.append({
                "damage_id":          f"D{did}",
                "damage_type":        "dent",
                "width_px":           w,
                "height_px":          h,
                "area_px":            w * h,
                "aspect_ratio":       round(w / max(h, 1), 2),
                "confidence":         1.0,
                "estimated_depth_mm": round(depth, 2),
            })
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            did += 1

    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return damage_list


# ── JSON OUTPUT BUILDER ───────────────────────────────────────────────────────

_SEVERITY_MAP = {
    ("High",   False): "high",
    ("High",   True):  "critical",
    ("Medium", False): "medium",
    ("Medium", True):  "high",
    ("Low",    False): "low",
    ("Low",    True):  "medium",
}

_DEPTH_LEVEL_MAP = {
    0:    {"level": "surface",          "detail": "Surface-level deformation only. No through-thickness penetration."},
    0.5:  {"level": "paint_and_primer", "detail": "Penetrates through topcoat. Primer layer exposed."},
    1.0:  {"level": "bare_metal",       "detail": "Full paint and primer removal. Bare substrate exposed."},
    99.0: {"level": "through_thickness","detail": "Suspected through-thickness defect. NDT required."},
}

def _depth_level(depth_mm: float) -> dict:
    if depth_mm == 0:
        return _DEPTH_LEVEL_MAP[0]
    elif depth_mm < 0.5:
        return _DEPTH_LEVEL_MAP[0.5]
    elif depth_mm < 2.0:
        return _DEPTH_LEVEL_MAP[1.0]
    else:
        return _DEPTH_LEVEL_MAP[99.0]

def _downtime(severity: str) -> dict:
    table = {
        "low":      {"min_hours": 2,  "max_hours": 4,  "notes": "Cosmetic repair. Can be deferred to next scheduled check."},
        "medium":   {"min_hours": 3,  "max_hours": 8,  "notes": "Repair within 72 hours. Line maintenance slot sufficient."},
        "high":     {"min_hours": 6,  "max_hours": 12, "notes": "Prompt repair required. Temporary protection if deferred."},
        "critical": {"min_hours": 24, "max_hours": 72, "notes": "AIRCRAFT ON GROUND (AOG). Do not dispatch without sign-off."},
    }
    entry = table.get(severity, table["medium"])
    return {**entry, "unit": "hours"}

def build_json_report(
    llm_report:  RepairRecommendationOutput,
    damage_list: list,
    meta:        dict,
) -> dict:
    """Maps LLM structured output + raw detections into the target JSON schema."""

    # Index raw detections by damage_id for dimension lookup
    raw = {d["damage_id"]: d for d in damage_list}

    damages_out = []
    for dmg in llm_report.damages:
        did  = dmg.damage_id
        raw_d = raw.get(did, {})
        depth_mm = raw_d.get("estimated_depth_mm", 0)
        dtype    = raw_d.get("damage_type", "unknown")
        w_px     = raw_d.get("width_px",  0)
        h_px     = raw_d.get("height_px", 0)

        severity = _SEVERITY_MAP.get(
            (dmg.confidence_level, dmg.escalation_required), "medium"
        )

        # Build the two repair recommendation objects expected by the schema
        recs = [
            {
                "confidence": {"High": 0.90, "Medium": 0.70, "Low": 0.50}[dmg.confidence_level],
                "text": dmg.justification,
            }
        ]
        if dmg.escalation_required:
            recs.append({
                "confidence": 0.60,
                "text": "Escalate to MRO / structural engineer before any repair attempt.",
            })

        damages_out.append({
            "damage_id":   did,
            "damage_type": dtype,
            "description": dmg.geometry_summary,
            "detailed_description": (
                f"{dmg.risk_assessment} {dmg.justification}"
            ),
            "location": "Detected via video frame analysis",
            "dimensions": {
                "width":  str(w_px),
                "height": str(h_px),
                "depth":  str(depth_mm),
                "unit":   "px/mm (width & height in px, depth in mm)",
            },
            "depth_estimate": _depth_level(depth_mm),
            "downtime_estimate": _downtime(severity),
            "severity": severity,
            "repair_recommendations": recs,
        })

    # Prioritization → sorted list
    pri_sorted = sorted(llm_report.prioritization, key=lambda x: x.priority_rank)

    s = llm_report.overall_repair_strategy
    output = {
        **meta,
        "bird_strike_detected": {
            "detected":       False,
            "confidence":     None,
            "affected_zones": [],
            "notes": "Bird-strike analysis not performed in video pipeline.",
        },
        "damages": damages_out,
        "overall_repair_strategy": {
            "general_repair_approach":     s.general_repair_approach,
            "interaction_between_damages": s.interaction_between_damages,
            "inspection_requirements":     s.inspection_requirements,
            "structural_risk_level":       s.structural_risk_level,
            "engineering_review_required": s.engineering_review_required,
        },
        "prioritization": [
            {
                "damage_id":    p.damage_id,
                "priority_rank": p.priority_rank,
                "justification": p.justification,
            }
            for p in pri_sorted
        ],
    }
    return output


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def process_video(video_path: str, meta: dict | None = None, output_json: str = "report.json"):
    try:
        if meta is None:
            meta = DEFAULT_META.copy()

        # 1. Pick best frame
        print(f"\n📹 Processing video: {video_path}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        frame = get_best_frame(video_path)
        if frame is None:
            print("❌ No usable frame found in video.")
            return
        print("✓ Best frame extracted")

        # 2. Detect damage
        print("\n🔍 Detecting damage in frame...")
        damage_list = detect_damage_frame(frame)
        if not damage_list:
            print("❌ No damage detected.")
            return
        print(f"✓ Detected {len(damage_list)} damage(s)")

        print(f"\n🤖 Sending to LLM for analysis...\n")

        # 3. LLM repair recommendation
        llm_report = get_repair_recommendation(damage_list)
        print("✓ LLM analysis complete")

        # 4. Build JSON output
        print("\n📋 Building repair report...")
        report_json = build_json_report(llm_report, damage_list, meta)
        print("✓ Report built")

        # 5. Save JSON
        try:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(report_json, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Report saved → {output_json}")
        except Exception as e:
            print(f"❌ ERROR saving report: {e}")
            raise

        # 6. Pretty-print to console
        print(json.dumps(report_json, indent=2))

        return report_json
    
    except FileNotFoundError as e:
        print(f"\n❌ FILE NOT FOUND: {e}")
        return None
    except Exception as e:
        print(f"\n❌ ERROR processing video: {e}")
        import traceback
        traceback.print_exc()
        return None


# ── CLI ENTRY POINT ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="SmartEye – Aircraft Damage Reporter")
        parser.add_argument("--video",    required=True,             help="Path to input video file")
        parser.add_argument("--output",   default="report.json",     help="Output JSON file path")
        parser.add_argument("--location", default="UNKNOWN",         help="Airport / location code")
        parser.add_argument("--reg",      default="UNKNOWN",         help="Aircraft registration (e.g. F-GKXA)")
        parser.add_argument("--aircraft", default="UNKNOWN",         help="Aircraft type (e.g. Airbus A320-214)")
        parser.add_argument("--airline",  default="UNKNOWN",         help="Airline name")
        args = parser.parse_args()

        meta = {
            "session_id":  DEFAULT_META["session_id"],
            "report_date": DEFAULT_META["report_date"],
            "location":    args.location,
            "aircraft": {
                "registration": args.reg,
                "type":         args.aircraft,
                "airline":      args.airline,
            },
        }

        process_video(args.video, meta=meta, output_json=args.output)
    except KeyboardInterrupt:
        print("\n✖ Process interrupted by user.")
    except Exception as e:
        print(f"\n✖ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()