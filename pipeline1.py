import cv2
import torch
import torch.nn as nn
import numpy as np
import joblib
from ultralytics import YOLO
from torchvision import models, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.stats import kurtosis

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# LOAD SCALER
# =========================
scaler = joblib.load("depth_scaler.pkl")

# =========================
# LOAD SINGLE YOLO MODEL
# =========================
# Assumed class mapping:
# 0 → dent
# 1 → scratch
# Change if your dataset differs
yolo_model = YOLO("yolo_v11.pt")
yolo_model.to(device)

DENT_CLASS = 1
SCRATCH_CLASS = 0

# =========================
# DEPTH REGRESSOR
# =========================
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
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.regressor(self.backbone(x))

depth_model = DentRegressor().to(device)
depth_model.load_state_dict(torch.load("best_model.pth", map_location=device))
depth_model.eval()

# =========================
# AUX FEATURES
# =========================
def compute_aux_labels(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)

    bright = int(np.mean(v > 230) > 0.02)
    specular = int(np.mean((v > 240) & (grad_mag < grad_mag.mean())) > 0.005)

    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).flatten()
    hist /= hist.sum() + 1e-6
    sharp_peak = int(kurtosis(hist) > 8)

    depth_grad = int(np.var(grad_mag) > 50)

    h,w = gray.shape
    yy,xx = np.indices((h,w))
    r = np.sqrt((xx-w/2)**2 + (yy-h/2)**2)
    r /= r.max() + 1e-6
    corr = np.corrcoef(r.flatten(), gray.flatten())[0,1]
    radial = int(abs(corr) > 0.15)

    return (bright, specular, sharp_peak, depth_grad, radial)

# =========================
# DENT CLASSIFIER
# =========================
class DentResNetWithAux(nn.Module):
    def __init__(self, num_aux=5):
        super().__init__()
        self.backbone = models.resnet18(weights="IMAGENET1K_V1")
        self.backbone.fc = nn.Identity()

        self.fc_img = nn.Linear(512, 128)
        self.fc_aux = nn.Sequential(
            nn.Linear(num_aux, 32),
            nn.ReLU()
        )
        self.fc_out = nn.Linear(160, 1)

    def forward(self, img, aux):
        img_feat = self.fc_img(self.backbone(img))
        aux_feat = self.fc_aux(aux)
        return self.fc_out(torch.cat([img_feat, aux_feat], dim=1))

dent_classifier = DentResNetWithAux().to(device)
dent_classifier.load_state_dict(torch.load("yes_no_model.pth", map_location=device))
dent_classifier.eval()

# =========================
# TRANSFORMS
# =========================
depth_tfms = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=(0.485,0.456,0.406),
                std=(0.229,0.224,0.225)),
    ToTensorV2()
])

clf_tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# =========================
# HELPERS
# =========================
@torch.no_grad()
def is_real_dent(crop):
    img = clf_tfms(crop).unsqueeze(0).to(device)

    aux = torch.tensor(
        compute_aux_labels(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)),
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    prob = torch.sigmoid(dent_classifier(img, aux)).item()
    return prob >= 0.6, prob

@torch.no_grad()
def predict_depth(crop):
    t = depth_tfms(image=crop)["image"].unsqueeze(0).to(device)
    pred = depth_model(t).cpu().numpy()
    return scaler.inverse_transform(pred)[0][0]

# =========================
# REALTIME PIPELINE
# =========================
def run_realtime(cam_id=0, skip_frames=2, conf=0.25):
    cap = cv2.VideoCapture(cam_id)
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame_id % skip_frames == 0:

            results = yolo_model.predict(
                source=rgb,
                conf=conf,
                device=0 if device.type == "cuda" else "cpu",
                verbose=False
            )[0]

            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()

                for box, cls in zip(boxes, classes):
                    x1,y1,x2,y2 = map(int, box)
                    cls = int(cls)

                    # ---------- SCRATCH ----------
                    if cls == SCRATCH_CLASS:
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                        cv2.putText(frame,"Scratch",(x1,y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

                    # ---------- DENT ----------
                    elif cls == DENT_CLASS:

                        if (x2-x1) < 40 or (y2-y1) < 40:
                            continue

                        crop = rgb[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue

                        # validate dent
                        ok, prob = is_real_dent(crop)
                        if not ok:
                            continue

                        # predict depth
                        depth = predict_depth(crop)

                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.putText(frame,
                                    f"Dent {depth:.2f}mm p={prob:.2f}",
                                    (x1,y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        cv2.imshow("Dent + Scratch Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =========================
# RUN
# =========================
if __name__ == "__main__":
    run_realtime(cam_id=0)
