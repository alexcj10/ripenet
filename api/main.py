import os
import sys
import torch
import random
import io
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms

# Add root directory to sys.path to allow imports from src, src_identity, src_regression
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.model import FruitRipenessModel
from src_regression.model import FruitRegressionModel
from src_identity.model import FruitIdentityModel

app = FastAPI(title="RipeNet API", description="AI-powered fruit species identification, ripeness classification, and shelf-life estimation.")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- CONFIG --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_MODEL_PATH = os.path.join(ROOT_DIR, "saved_models", "best_model.pth")
REGRESS_MODEL_PATH = os.path.join(ROOT_DIR, "saved_models", "best_regression_model.pth")
IDENTITY_MODEL_PATH = os.path.join(ROOT_DIR, "saved_models", "best_identity_model.pth")

STAGE_LABELS = {0: "unripe", 1: "fresh", 2: "rotten"}
FRUIT_LABELS = {0: "apple", 1: "banana", 2: "orange", 3: "papaya"}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Reuse TEMPLATES from predict.py (abbreviated here, or just import it if possible)
# For the sake of robustness and speed, I'll copy a few or try to import from predict.py
try:
    from src.predict import TEMPLATES
except ImportError:
    # Minimal fallback templates if import fails
    TEMPLATES = {
        "apple": {"unripe": ["Unripe apple. Needs {days:.1f} days."], "fresh": ["Fresh apple. Good for {days:.1f} days."], "rotten": ["Rotten apple."]},
        "banana": {"unripe": ["Unripe banana. Needs {days:.1f} days."], "fresh": ["Fresh banana. Good for {days:.1f} days."], "rotten": ["Rotten banana."]},
        "orange": {"unripe": ["Unripe orange. Needs {days:.1f} days."], "fresh": ["Fresh orange. Good for {days:.1f} days."], "rotten": ["Rotten orange."]},
        "papaya": {"unripe": ["Unripe papaya. Needs {days:.1f} days."], "fresh": ["Fresh papaya. Good for {days:.1f} days."], "rotten": ["Rotten papaya."]},
        "unknown": {"unripe": ["Unripe fruit."], "fresh": ["Fresh fruit."], "rotten": ["Rotten fruit."]}
    }

# Global variables for models
class_model = None
reg_model = None
id_model = None

def load_models():
    global class_model, reg_model, id_model
    
    print(f"Loading models to {DEVICE}...")
    
    # Load Classification (Stage)
    class_model = FruitRipenessModel(num_classes=3)
    class_model.load_state_dict(torch.load(CLASS_MODEL_PATH, map_location=DEVICE))
    class_model.to(DEVICE)
    class_model.eval()

    # Load Regression (Days)
    reg_model = FruitRegressionModel()
    reg_model.load_state_dict(torch.load(REGRESS_MODEL_PATH, map_location=DEVICE))
    reg_model.to(DEVICE)
    reg_model.eval()

    # Load Identity (Fruit Type)
    id_model = FruitIdentityModel(num_classes=4)
    if os.path.exists(IDENTITY_MODEL_PATH):
        id_model.load_state_dict(torch.load(IDENTITY_MODEL_PATH, map_location=DEVICE))
        id_model.to(DEVICE)
        id_model.eval()
    else:
        print("⚠️ Identity model not found.")
        id_model = None

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
            # 1. Identity
            fruit_name = "unknown"
            if id_model:
                id_output = id_model(input_tensor)
                fruit_idx = id_output.argmax(dim=1).item()
                fruit_name = FRUIT_LABELS[fruit_idx]
            
            # 2. Ripeness Stage
            class_output = class_model(input_tensor)
            stage_idx = class_output.argmax(dim=1).item()
            stage = STAGE_LABELS[stage_idx]
            
            # 3. Shelf Life
            reg_output = reg_model(input_tensor)
            days = reg_output.item()
            
        # Format the result
        display_days = max(0, days) if stage != "rotten" else abs(days)
        
        # Pick template
        templates_for_fruit = TEMPLATES.get(fruit_name, TEMPLATES["unknown"])
        sentence_list = templates_for_fruit.get(stage, TEMPLATES["unknown"][stage])
        sentence = random.choice(sentence_list).format(days=display_days)
        
        return {
            "fruit": fruit_name.capitalize(),
            "ripeness": stage.capitalize(),
            "shelf_life_days": round(display_days, 2),
            "report": sentence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
