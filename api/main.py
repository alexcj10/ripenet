import os
import torch
import random
import io
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms, models
import torch.nn as nn

# Local imports
import sys
# Ensure we can import from the root directory so ml module is found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ml.core.multi_task_model import RipeNetMTL

app = FastAPI(title="RipeNet API", description="AI-powered fruit ripeness detection (Multi-Task V2)")

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# New unified model path
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "saved_models", "ripenet_v2_mtl.pth"))

# Labels (Synced with train.py)
STAGE_LABELS = {0: "unripe", 1: "fresh", 2: "rotten"}
FRUIT_LABELS = {
    0: "apple", 
    1: "banana", 
    2: "mango", 
    3: "orange", 
    4: "papaya", 
    5: "pineapple"
}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------- TEMPLATES --------
TEMPLATES = {
    "apple": {
        "unripe": [
            "This apple is still quite green and needs about {days:.1f} days to sweeten up.",
            "Patience is key! This apple is unripe. Give it another {days:.1f} days.",
            "A bit tart for now—it's still unripe. Check back in {days:.1f} days.",
            "Too crunchy and sour. It needs {days:.1f} days of ripening time.",
            "Not quite ready for a bite. Status: Unripe. Window: {days:.1f} days.",
            "A young apple indeed! Let it mature for {days:.1f} more days.",
            "Wait for that characteristic sweetness. Unripe for {days:.1f} days.",
            "Firm and green. This one needs {days:.1f} days to reach peak flavor.",
            "Not quite dessert-ready. It's unripe. Estimated wait: {days:.1f} days.",
            "The sugars are still developing. Give it {days:.1f} days."
        ],
        "fresh": [
            "Perfect timing! This apple is fresh and at its peak for the next {days:.1f} days.",
            "Crunchy, sweet, and fresh. Enjoy it within the next {days:.1f} days.",
            "An apple a day! This one is fresh for {days:.1f} days.",
            "Look at that shine! It's perfectly fresh for another {days:.1f} days.",
            "Optimal crispness detected. Fresh for {days:.1f} more days.",
            "Top-tier quality fruit. Status: Fresh. Lasting {days:.1f} days.",
            "Ready to be enjoyed! It will stay fresh for roughly {days:.1f} days.",
            "Juicy and firm. This fresh apple is good for {days:.1f} days.",
            "A delicious snack awaits! Fresh and shelf-stable for {days:.1f} days.",
            "Beautifully ripe. Enjoy this fresh apple within {days:.1f} days."
        ],
        "rotten": [
            "Unfortunately, this apple is rotten. It's {days:.1f} days past its prime.",
            "This one is a goner—it's rotten. Probably should have been eaten {days:.1f} days ago.",
            "🚨 WARNING: This apple is rotten and unsafe.",
            "Spoilage detected. Status: Rotten. Expired {days:.1f} days back.",
            "Too late for this one. It's rotten by about {days:.1f} days.",
            "Mushy and unappealing. This apple is rotten and {days:.1f} days old.",
            "Decomposition has started. It's rotten. Past due: {days:.1f} days.",
            "Not for human consumption. Status: Rotten. Age: {days:.1f} days.",
            "Nature has taken its course. This apple is rotten.",
            "Expired fruit alert. This one is rotten and {days:.1f} days past fresh."
        ]
    },
    "banana": {
        "unripe": [
            "This banana is still green! Give it about {days:.1f} days to turn yellow.",
            "Too starchy to eat now. It's unripe. Wait {days:.1f} days.",
            "A very firm, green banana. Unripe for {days:.1f} more days.",
            "Wait for the sugar to kick in. Status: Unripe. Time: {days:.1f} days.",
            "Not quite ready to peel. It's unripe. Give it {days:.1f} days.",
            "Keep it in a paper bag! Unripe for another {days:.1f} days.",
            "Still maturing. This banana needs {days:.1f} days to ripen.",
            "Hard as a rock! Let it sit for {days:.1f} days.",
            "The countdown to yellow is {days:.1f} days. Status: Unripe.",
            "No banana bread yet—it's still unripe for {days:.1f} days."
        ],
        "fresh": [
            "Perfect yellow! This banana is fresh and good for {days:.1f} days.",
            "Sweet and creamy. It's fresh for another {days:.1f} days.",
            "Just enough spots for sweetness! Fresh for {days:.1f} more days.",
            "Ready for your cereal! Status: Fresh. Lasting {days:.1f} days.",
            "Ideal ripeness reached. Enjoy this fresh fruit for {days:.1f} days.",
            "Vibrant yellow detected. Good and fresh for {days:.1f} days.",
            "Sweet treats incoming! It's fresh for the next {days:.1f} days.",
            "A healthy snack ready now. Fresh for about {days:.1f} days.",
            "Top quality banana. Status: Fresh. Window: {days:.1f} days.",
            "Enjoy at its peak! Fresh for {days:.1f} more days."
        ],
        "rotten": [
            "This banana has seen better days. It's rotten and {days:.1f} days old.",
            "Mushy and dark—it's rotten. Overdue by {days:.1f} days.",
            "🚨 SPOILED: This banana is rotten and far past its prime.",
            "Maybe good for baking? But status is: Rotten for {days:.1f} days.",
            "Too soft for comfort. It's rotten. Expired {days:.1f} days ago.",
            "Throw it out! It's rotten and {days:.1f} days past fresh.",
            "Fermentation detected. Status: Rotten. Past due: {days:.1f} days.",
            "Black all over. This banana is rotten by {days:.1f} days.",
            "The shelf life ended {days:.1f} days ago. It's rotten.",
            "Avoid this mushy mess. Status: Rotten. Age: {days:.1f} days."
        ]
    },
    "mango": {
        "unripe": [
            "This mango is hard and sour. It needs {days:.1f} days to reach peak sweetness.",
            "Wait for that tropical aroma! It's currently unripe for {days:.1f} days.",
            "Firm and green patches. Give it {days:.1f} days to soften.",
            "Patience for the king of fruits! Unripe for {days:.1f} days.",
            "Not juicy yet. Status: Unripe. Estimated wait: {days:.1f} days.",
            "Wait for the golden glow. It's unripe for {days:.1f} more days.",
            "Still quite tart. Let it ripen for {days:.1f} days.",
            "A sturdy mango that needs {days:.1f} days of patience.",
            "Tropical treat in progress. Unripe for {days:.1f} days.",
            "Give it some warmth. It's unripe for another {days:.1f} days."
        ],
        "fresh": [
            "A perfect, juicy mango! Enjoy its sweetness over the next {days:.1f} days.",
            "Status: Fresh. This mango is at its peak for about {days:.1f} days.",
            "Exquisite aroma detected. Fresh and ready for {days:.1f} days.",
            "Beautifully ripe! Enjoy it within the next {days:.1f} days.",
            "Sweet and yielding. This fresh mango is good for {days:.1f} days.",
            "The ultimate tropical snack. Fresh for {days:.1f} more days.",
            "Ready to be sliced! Status: Fresh. Lasting {days:.1f} days.",
            "Top tier ripeness. Enjoy this fresh fruit for {days:.1f} days.",
            "Vibrant and delicious. Fresh for the next {days:.1f} days.",
            "Nature's candy is ready! Fresh for {days:.1f} more days."
        ],
        "rotten": [
            "The mango has fermented. It's rotten and {days:.1f} days past edible.",
            "🚨 WARNING: Spoilage detected. This mango is {days:.1f} days expired.",
            "Too soft and smelling sour. It's rotten for {days:.1f} days.",
            "Mushy patches found. This mango is rotten.",
            "Not fit for eating. Status: Rotten. Expired {days:.1f} days ago.",
            "Decomposition alert! It's rotten by about {days:.1f} days.",
            "Throw it in the compost. It's rotten and {days:.1f} days old.",
            "The tropical fruit has turned. Status: Rotten. Past due: {days:.1f} days.",
            "Avoid this one. It's rotten and {days:.1f} days past its prime.",
            "Spoiled beyond rescue. Status: Rotten."
        ]
    },
    "orange": {
        "unripe": [
            "This orange is still a bit sour. Give it {days:.1f} days to sweeten.",
            "Not quite juicy yet. Status: Unripe. Window: {days:.1f} days.",
            "Firm skin and green patches. Wait {days:.1f} days.",
            "Too acidic right now. It's unripe. Give it {days:.1f} days.",
            "Let the sugars develop! Unripe for {days:.1f} more days.",
            "A bit tart for juice. It's unripe for {days:.1f} days.",
            "Wait for the orange glow. Status: Unripe. Time: {days:.1f} days.",
            "Still maturing on the counter. Give it {days:.1f} days.",
            "Not quite dessert-ready. Unripe for {days:.1f} more days.",
            "Patience for that citrus zing! Unripe for {days:.1f} days."
        ],
        "fresh": [
            "Zesty and sweet! This fresh orange will last {days:.1f} days.",
            "Perfect for juice! It's fresh and good for {days:.1f} more days.",
            "Full of Vitamin C! Enjoy it over the next {days:.1f} days.",
            "Vibrant and firm. This fresh orange has {days:.1f} days left.",
            "Look at that color! It's fresh for another {days:.1f} days.",
            "A refreshing snack ready now. Fresh for {days:.1f} days.",
            "Ready to be peeled. Status: Fresh. Lasting {days:.1f} days.",
            "Top tier citrus. Enjoy this fresh fruit for {days:.1f} days.",
            "Healthy and delicious. Fresh for {days:.1f} more days.",
            "Sweetness detected! Best used in the next {days:.1f} days."
        ],
        "rotten": [
            "This orange has gone bad. It's rotten and {days:.1f} days past due.",
            "Mold alert! It's rotten. It expired about {days:.1f} days ago.",
            "The citrus has spoiled. Status: Rotten. Age: {days:.1f} days.",
            "Do not consume! This orange is rotten and old by {days:.1f} days.",
            "Mushy patches detected. It's rotten for {days:.1f} days.",
            "Nature is breaking it down. Status: Rotten. Past: {days:.1f} days.",
            "Throw it out! It's rotten and {days:.1f} days past fresh.",
            "Expired fruit alert. This orange is rotten by {days:.1f} days.",
            "Not safe to eat. Status: Rotten. Age: {days:.1f} days.",
            "The shelf life ended {days:.1f} days ago. It's rotten."
        ]
    },
    "papaya": {
        "unripe": [
            "This papaya is very firm. Give it {days:.1f} days to soften.",
            "Tropical treat in progress... Unripe for {days:.1f} more days.",
            "Wait for the skin to yellow! Unripe for {days:.1f} days.",
            "Too hard for breakfast. Give it {days:.1f} days of ripening.",
            "Not sweet yet. Status: Unripe. Estimated wait: {days:.1f} days.",
            "Still green outside. It's unripe for {days:.1f} more days.",
            "Let it soften up. Unripe for {days:.1f} days.",
            "A bit too sturdy. Let it mature for {days:.1f} days.",
            "Patience for the papaya! Unripe for {days:.1f} more days.",
            "Wait for that orange glow. Currently unripe for {days:.1f} days."
        ],
        "fresh": [
            "Soft and sweet! This fresh papaya is good for {days:.1f} days.",
            "Beautifully ripe papaya. Enjoy it within the next {days:.1f} days.",
            "Perfect tropical treat! Ready for you for {days:.1f} days.",
            "Yielding to pressure. Status: Fresh for {days:.1f} more days.",
            "Top tier fruit alert. Enjoy this fresh papaya for {days:.1f} days.",
            "Savor it today! It's fresh and good for {days:.1f} more days.",
            "Rich orange color detected. Fresh for {days:.1f} days.",
            "Ready for slicing. Status: Fresh. Lasting {days:.1f} days.",
            "Ideal ripeness reached. Enjoy for the next {days:.1f} days.",
            "Delicious and ready! Fresh for about {days:.1f} days."
        ],
        "rotten": [
            "This papaya has spoiled. It's rotten and {days:.1f} days past due.",
            "🚨 WARNING: Not fit for eating. Status: Rotten.",
            "Very mushy and smelly. It's rotten. Expired {days:.1f} days ago.",
            "Spoilage detected near the stem. It's rotten for {days:.1f} days.",
            "Throw it away! It's rotten and {days:.1f} days past fresh.",
            "The papaya has fermented. Status: Rotten. Age: {days:.1f} days.",
            "Avoid this one completely. It's rotten.",
            "Leaking juices found. It's rotten by about {days:.1f} days.",
            "Not for human consumption. Status: Rotten. Expired {days:.1f} days ago.",
            "Total decay identified. It's rotten and {days:.1f} days old."
        ]
    },
    "pineapple": {
        "unripe": [
            "Too acidic right now! This pineapple needs {days:.1f} days to sweeten up.",
            "The base isn't yellow yet. It's unripe for {days:.1f} more days.",
            "Wait for the smell of paradise! Unripe for {days:.1f} days.",
            "A bit too green. Let it ripen for {days:.1f} more days.",
            "Not quite juicy enough. Status: Unripe. Target: {days:.1f} days.",
            "A tough pineapple that needs {days:.1f} days of patience.",
            "Let the sugars climb the fruit! Unripe for {days:.1f} days.",
            "Wait for those bottom leaves to yield. Unripe for {days:.1f} days.",
            "Not sweet yet. Status: Unripe. Estimated time: {days:.1f} days.",
            "The countdown to gold is {days:.1f} days. Status: Unripe."
        ],
        "fresh": [
            "Perfectly sweet and tart! Enjoy this fresh pineapple for {days:.1f} days.",
            "Smells like paradise! Status: Fresh. Lasting {days:.1f} days.",
            "Vibrant yellow base detected. Fresh for {days:.1f} more days.",
            "Ideal tropical snack! Enjoy it within the next {days:.1f} days.",
            "Sweet aroma and firm leaves. Fresh for {days:.1f} days.",
            "Ready for your fruit salad! Status: Fresh for {days:.1f} days.",
            "Deliciously ripe. Enjoy this fresh fruit for {days:.1f} more days.",
            "Top tier pineapple alert. Fresh for the next {days:.1f} days.",
            "Grab the knife! It's fresh and good for {days:.1f} days.",
            "Tropical perfection. Status: Fresh. Window: {days:.1f} days."
        ],
        "rotten": [
            "The pineapple has soured. It's rotten and {days:.1f} days past its prime.",
            "🚨 WARNING: Fermentation detected. This pineapple is rotten.",
            "Too soft and leaking. It's rotten for {days:.1f} days.",
            "The base looks dark and spoiled. Status: Rotten.",
            "Mushy patches and off-odor. It's rotten by about {days:.1f} days.",
            "Do not consume! Status: Rotten. Expired {days:.1f} days ago.",
            "Nature is reclaiming this one. It's rotten.",
            "Throw it out! It's rotten and {days:.1f} days past fresh.",
            "Spoilage identified. Not safe. Status: Rotten.",
            "The shelf life ended {days:.1f} days ago. It's rotten."
        ]
    },
    "unknown": {
        "unripe": ["This item appears to be unripe. Estimated time: {days:.1f} days."],
        "fresh": ["This item looks fresh! Good for another {days:.1f} days."],
        "rotten": ["This item is rotten. It was fresh {days:.1f} days ago."]
    }
}

# Global models
mtl_model = None

def load_models():
    global mtl_model
    print("🔄 Loading Multi-Task Model weights...")
    
    try:
        mtl_model = RipeNetMTL()
        if not os.path.exists(MODEL_PATH):
             # Fallback to check HF models dir
             alt_path = os.path.join(BASE_DIR, "models", "ripenet_v2_mtl.pth")
             final_path = alt_path if os.path.exists(alt_path) else MODEL_PATH
             mtl_model.load_state_dict(torch.load(final_path, map_location=DEVICE))
        else:
            mtl_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            
        mtl_model.to(DEVICE).eval()
        print("✅ RipeNet 2.0 Multi-Task Model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/")
async def root():
    return {"message": "RipeNet 2.0 API is running", "status": "online"}

@app.post("/predict")
async def predict_ripeness(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        input_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = mtl_model(input_tensor)
            
            # Identity
            fruit_idx = outputs["fruit"].argmax(dim=1).item()
            fruit_name = FRUIT_LABELS.get(fruit_idx, "unknown")
            
            # Stage
            stage_idx = outputs["ripeness"].argmax(dim=1).item()
            stage = STAGE_LABELS.get(stage_idx, "unknown")
            
            # Shelf life
            days = outputs["days"].item()
            
        display_days = max(0, days) if stage != "rotten" else abs(days)
        templates_for_fruit = TEMPLATES.get(fruit_name, TEMPLATES["unknown"])
        sentence_list = templates_for_fruit.get(stage, TEMPLATES["unknown"].get(stage, ["Analysis complete."]))
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
