import os
import torch
import random
import io
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms, models
import torch.nn as nn

app = FastAPI(title="RipeNet API", description="AI-powered fruit ripeness detection")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- CONFIG --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {DEVICE}")

# Determine base directory
# If running as 'python api/main.py', __file__ is in the 'api' folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Models can be in 'api/models' (HF) or '../saved_models' (Local)
HF_MODELS = os.path.join(BASE_DIR, "models")
LOCAL_MODELS = os.path.abspath(os.path.join(BASE_DIR, "..", "saved_models"))

def find_model(name):
    paths = [
        os.path.join(HF_MODELS, name),
        os.path.join(LOCAL_MODELS, name)
    ]
    for p in paths:
        if os.path.exists(p):
            print(f"🔍 Found model: {name} at {p}")
            return p
    print(f"❌ Could not find model: {name}")
    return paths[0] # Fallback to first one (HF) so it shows in error

CLASS_MODEL_PATH = find_model("best_model.pth")
REGRESS_MODEL_PATH = find_model("best_regression_model.pth")
IDENTITY_MODEL_PATH = find_model("best_identity_model.pth")

STAGE_LABELS = {0: "unripe", 1: "fresh", 2: "rotten"}
FRUIT_LABELS = {0: "apple", 1: "banana", 2: "orange", 3: "papaya"}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------- MODEL DEFINITIONS --------
class FruitRipenessModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    def forward(self, x): return self.backbone(x)

class FruitRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.backbone(x)

class FruitIdentityModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )
    def forward(self, x): return self.backbone(x)

# -------- TEMPLATES --------
TEMPLATES = {
    "apple": {
        "unripe": ["This apple is still quite green and needs about {days:.1f} days to sweeten up."],
        "fresh": ["Perfect timing! This apple is fresh and at its peak for the next {days:.1f} days."],
        "rotten": ["Unfortunately, this apple is past its prime."]
    },
    "banana": {
        "unripe": ["This banana is still green and needs about {days:.1f} days to ripen."],
        "fresh": ["This banana is perfectly ripe! Enjoy it within {days:.1f} days."],
        "rotten": ["This banana is overripe and should be used for baking."]
    },
    "orange": {
        "unripe": ["This orange needs about {days:.1f} more days to fully ripen."],
        "fresh": ["This orange is fresh and juicy! Good for {days:.1f} days."],
        "rotten": ["This orange is past its prime."]
    },
    "papaya": {
        "unripe": ["This papaya needs about {days:.1f} days to ripen properly."],
        "fresh": ["This papaya is perfectly ripe! Best consumed within {days:.1f} days."],
        "rotten": ["This papaya is overripe."]
    },
    "unknown": {
        "unripe": ["This fruit needs more time to ripen."],
        "fresh": ["This fruit is fresh!"],
        "rotten": ["This fruit is past its prime."]
    }
}

# Global models
class_model = reg_model = id_model = None

def load_models():
    global class_model, reg_model, id_model
    print("🔄 Loading model weights...")
    
    try:
        class_model = FruitRipenessModel(3)
        class_model.load_state_dict(torch.load(CLASS_MODEL_PATH, map_location=DEVICE))
        class_model.to(DEVICE).eval()
        
        reg_model = FruitRegressionModel()
        reg_model.load_state_dict(torch.load(REGRESS_MODEL_PATH, map_location=DEVICE))
        reg_model.to(DEVICE).eval()
        
        id_model = FruitIdentityModel(4)
        id_model.load_state_dict(torch.load(IDENTITY_MODEL_PATH, map_location=DEVICE))
        id_model.to(DEVICE).eval()
        print("✅ All models loaded successfully!")
    except Exception as e:
        print(f"⚠️ Error loading models: {e}")

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/")
async def root():
    return {"message": "RipeNet API is running", "status": "online"}

@app.post("/predict")
async def predict_ripeness(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        input_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            # Identity
            id_output = id_model(input_tensor)
            fruit_idx = id_output.argmax(dim=1).item()
            fruit_name = FRUIT_LABELS.get(fruit_idx, "unknown")
            
            # Stage
            class_output = class_model(input_tensor)
            stage_idx = class_output.argmax(dim=1).item()
            stage = STAGE_LABELS.get(stage_idx, "unknown")
            
            # Shelf life
            reg_output = reg_model(input_tensor)
            days = reg_output.item()
            
        display_days = max(0, days) if stage != "rotten" else abs(days)
        templates_for_fruit = TEMPLATES.get(fruit_name, TEMPLATES["unknown"])
        sentence_list = templates_for_fruit.get(stage, TEMPLATES["unknown"].get(stage, ["Fruit analysis complete."]))
        sentence = random.choice(sentence_list).format(days=display_days)
        
        return {
            "fruit": fruit_name.capitalize(),
            "ripeness": stage.capitalize(),
            "shelf_life_days": round(display_days, 2),
            "report": sentence
        }
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
