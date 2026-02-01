import os
import torch
import random
from PIL import Image
from torchvision import transforms

# Import models from both directories
import sys
# Add current and regression paths to sys.path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from model import FruitRipenessModel # Classification (Stage)
from src_regression.model import FruitRegressionModel # Regression (Days)
from src_identity.model import FruitIdentityModel # Identity (Species)

# -------- CONFIG --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASS_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "best_model.pth")
REGRESS_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "best_regression_model.pth")
IDENTITY_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "best_identity_model.pth")

# Map stage indices to labels
STAGE_LABELS = {0: "unripe", 1: "fresh", 2: "rotten"}
# Map identity indices to fruit names
FRUIT_LABELS = {0: "apple", 1: "banana", 2: "orange", 3: "papaya"}

# Preprocessing (same as training)
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 🎭 MASSIVE TEMPLATE DICTIONARY (15+ variants per type)
TEMPLATES = {
    "apple": {
        "unripe": [
            "This apple is still quite green and needs about {days:.1f} days to sweeten up.",
            "Patience is key! This apple is unripe. Give it another {days:.1f} days.",
            "A bit tart for now—it's still unripe. Check back in {days:.1f} days.",
            "Not quite ready for a pie yet; it's still unripe. Estimated time: {days:.1f} days.",
            "This crunchy friend is still in its 'unripe' phase for about {days:.1f} more days.",
            "Crisp but sour. It's unripe and needs {days:.1f} days on the counter.",
            "Still maturing! This apple will be fresh in roughly {days:.1f} days.",
            "Its color gives it away—it's unripe. Estimated ripening time: {days:.1f} days.",
            "Hard as a rock! Let this unripe apple sit for {days:.1f} days.",
            "Too early for a snack. It's unripe. Wait {days:.1f} days for the best flavor.",
            "Nature is still working on this one. It's unripe for {days:.1f} more days.",
            "Give it some sun! This apple is unripe and needs {days:.1f} days.",
            "Still a 'baby' apple. It's unripe. Give it {days:.1f} days to grow up.",
            "Not sweet yet. This unripe apple has {days:.1f} days left to go.",
            "Hold off on picking! It's unripe and needs {days:.1f} days of patience.",
            "The model sees a very young, unripe apple. Check it in {days:.1f} days."
        ],
        "fresh": [
            "Perfect timing! This apple is fresh and at its peak for the next {days:.1f} days.",
            "Look at that shine! It's fresh and will stay that way for {days:.1f} days.",
            "Crunchy, sweet, and fresh. Enjoy it within the next {days:.1f} days.",
            "An apple a day! This one is fresh and prime for eating for {days:.1f} days.",
            "This fresh apple looks delicious. You've got {days:.1f} days to enjoy it.",
            "Top quality fruit here. It's fresh and good for another {days:.1f} days.",
            "Beautifully ripe! This fresh apple will last about {days:.1f} more days.",
            "Ready to be eaten! It's fresh and has a {days:.1f} day shelf life left.",
            "A prime specimen. This fresh apple is good for {days:.1f} days.",
            "Sweetness detected! It's fresh. Best used in the next {days:.1f} days.",
            "Doesn't get better than this. It's fresh for about {days:.1f} days.",
            "Solid and sweet. This fresh apple has {days:.1f} days of glory left.",
            "Juicy and ready! It's fresh. It should keep for {days:.1f} days.",
            "Eat up! This apple is perfectly fresh for another {days:.1f} days.",
            "The model flags this as a fresh, high-quality apple for {days:.1f} days.",
            "Vibrant and tasty. Status: Fresh. Window: {days:.1f} days."
        ],
        "rotten": [
            "Unfortunately, this apple is rotten. It's {days:.1f} days past its prime.",
            "This one is a goner—it's rotten. Probably should have been eaten {days:.1f} days ago.",
            "Spoilage detected. This rotten apple is roughly {days:.1f} days old.",
            "Not safe for eating. It's rotten and has been for {days:.1f} days.",
            "The texture looks soft—it's rotten. It expired about {days:.1f} days ago.",
            "Time to compost this one! It's rotten by about {days:.1f} days.",
            "Bruised and spoiled. This apple is rotten. Past due: {days:.1f} days.",
            "Uh oh, this apple is rotten. Total decay estimate: {days:.1f} days.",
            "Avoid this one; it's rotten. It's been bad for {days:.1f} days.",
            "Nature has taken it back. It's rotten. Overdue by {days:.1f} days.",
            "Fungal growth likely. Status: Rotten. Gone for {days:.1f} days.",
            "This doesn't look good. It's rotten and was fresh {days:.1f} days ago.",
            "Too late for this fruit. It's rotten. Expired {days:.1f} days back.",
            "The model identifies this as a rotten apple. Old by {days:.1f} days.",
            "Do not consume! This apple is rotten. Estimated age: {days:.1f} days."
        ]
    },
    "banana": {
        "unripe": [
            "This banana is still green! Give it about {days:.1f} days to turn yellow.",
            "Too starchy to eat now. It's unripe. Wait {days:.1f} days.",
            "A very young banana. Status: Unripe. Ripening in {days:.1f} days.",
            "Check back in {days:.1f} days; this banana is still unripe.",
            "Not quite 'peel-able' yet. It's unripe for {days:.1f} more days.",
            "Still maturing in the bunch. It's unripe for about {days:.1f} days.",
            "Keep it in a paper bag! This unripe banana needs {days:.1f} days.",
            "Tough skin and green tint. It's unripe. Give it {days:.1f} days.",
            "The countdown to yellow is {days:.1f} days. Status: Unripe.",
            "No smoothie today! It's unripe for another {days:.1f} days.",
            "Nature's wrapper is still green. Unripe for {days:.1f} days.",
            "Almost there, but still unripe. Check it in {days:.1f} days.",
            "Wait for the sugar to develop. Unripe for {days:.1f} more days.",
            "Firm and green. This banana is unripe. Target: {days:.1f} days.",
            "Predicted as unripe. It should be ready in {days:.1f} days."
        ],
        "fresh": [
            "Perfect yellow! This banana is fresh and good for {days:.1f} days.",
            "Ideal for your cereal. It's fresh and will stay that way for {days:.1f} days.",
            "Just enough spots! It's fresh. Eat it within {days:.1f} days.",
            "A beautiful fresh banana. You've got {days:.1f} days to enjoy it.",
            "Sweet and creamy. It's fresh for another {days:.1f} days.",
            "Best time to eat! Status: Fresh. Lasting for {days:.1f} days.",
            "Ready for peeling. This fresh banana has {days:.1f} days left.",
            "High energy snack! It's fresh for about {days:.1f} more days.",
            "Smooth and perfectly yellow. Fresh for {days:.1f} days.",
            "A great fresh find! Keep it for up to {days:.1f} days.",
            "Perfect for a lunchbox. It's fresh. Limit: {days:.1f} days.",
            "Vibrant yellow detected. Status: Fresh. Time: {days:.1f} days.",
            "Grab it now! It's fresh and good for {days:.1f} more days.",
            "Top tier banana here. Fresh for the next {days:.1f} days.",
            "Highly recommended. It's fresh. Enjoy within {days:.1f} days."
        ],
        "rotten": [
            "This banana has seen better days. It's rotten and {days:.1f} days old.",
            "Mushy and dark—it's rotten. Overdue by {days:.1f} days.",
            "Maybe good for banana bread? It's rotten by {days:.1f} days.",
            "Too soft for comfort. It's rotten. Expired {days:.1f} days ago.",
            "The model sees a rotten banana. Past its prime for {days:.1f} days.",
            "Compost time! This banana is rotten. Age: {days:.1f} days.",
            "Brown all over... it's rotten. Bad for {days:.1f} days.",
            "Do not slice this! It's rotten and old by {days:.1f} days.",
            "Fermentation might have started. Rotten for {days:.1f} days.",
            "Not safe to eat raw. It's rotten. Over: {days:.1f} days.",
            "Deep brown spots everywhere. Status: Rotten. Day {days:.1f}.",
            "Expired fruit alert! This banana is rotten. Past due: {days:.1f} days.",
            "The shelf life ended {days:.1f} days ago. It's rotten.",
            "Identified as rotten. It's been spoiled for {days:.1f} days.",
            "Throw it out! It's rotten and {days:.1f} days past fresh."
        ]
    },
    "orange": {
        "unripe": [
            "This orange is still a bit sour. Give it {days:.1f} days to sweeten.",
            "Too firm and bit green. It's unripe. Wait {days:.1f} days.",
            "Not quite juicy yet. Status: Unripe. Window: {days:.1f} days.",
            "The zest isn't ready. This unripe orange needs {days:.1f} days.",
            "Still sour! It's unripe and needs {days:.1f} more days.",
            "Let those sugars develop. Unripe for {days:.1f} days.",
            "Patience for the citrus... it's unripe for {days:.1f} days.",
            "Check back in {days:.1f} days for a juicy orange. Currently unripe.",
            "The model sees a green, unripe orange. Target: {days:.1f} days.",
            "Wait for that orange glow! Currently unripe for {days:.1f} days.",
            "A bit bitter right now. It's unripe. Days left: {days:.1f}.",
            "Give it some time to juice up. Unripe for {days:.1f} days.",
            "Not ready for squeezing. Status: Unripe. Wait {days:.1f} days.",
            "Tough skin and green patches. Unripe for {days:.1f} days.",
            "Predicted as immature. Status: Unripe. Time: {days:.1f} days."
        ],
        "fresh": [
            "Zesty and sweet! This fresh orange will last {days:.1f} days.",
            "Perfect for juice! It's fresh and good for {days:.1f} more days.",
            "Full of Vitamin C! This orange is fresh for {days:.1f} days.",
            "A vibrant fresh orange. Enjoy it within the next {days:.1f} days.",
            "Firm and juicy. It's fresh for another {days:.1f} days.",
            "Look at that color! Status: Fresh. Lasting for {days:.1f} days.",
            "Ready for peeling. This fresh orange has {days:.1f} days left.",
            "A refreshing citrus snack. Fresh for {days:.1f} more days.",
            "High quality citrus detected. Fresh for {days:.1f} days.",
            "Squeeze it today! It's fresh for up to {days:.1f} days.",
            "Beautifully bright. Status: Fresh. Window: {days:.1f} days.",
            "The model loves this fresh orange! Best in {days:.1f} days.",
            "Peak sweetness! Enjoy this fresh fruit for {days:.1f} days.",
            "Healthy and ready. It's fresh and good for {days:.1f} days.",
            "Top tier orange here! Status: Fresh for {days:.1f} days."
        ],
        "rotten": [
            "This orange has gone bad. It's rotten and {days:.1f} days past due.",
            "Mold alert! It's rotten. It expired about {days:.1f} days ago.",
            "Too soft and smelling sour. It's rotten for {days:.1f} days.",
            "The citrus has spoiled. Status: Rotten. Age: {days:.1f} days.",
            "Do not consume! This orange is rotten and old by {days:.1f} days.",
            "Mushy patches found. It's rotten. Expired {days:.1f} days ago.",
            "Nature is breaking it down. Status: Rotten. Past: {days:.1f} days.",
            "The model flags this as a spoiled orange. Old by {days:.1f} days.",
            "Throw it in the bin; it's rotten for {days:.1f} days.",
            "Life has left this fruit. Status: Rotten. Expired {days:.1f} days.",
            "Smell it? It's rotten. Total days past fresh: {days:.1f}.",
            "Discoloration and decay found. Rotten for {days:.1f} days.",
            "The shelf life ended {days:.1f} days ago. It's rotten.",
            "Identified as spoiled. It's been rotten for {days:.1f} days.",
            "Avoid at all costs! It's rotten and {days:.1f} days past its prime."
        ]
    },
    "papaya": {
        "unripe": [
            "This papaya is very firm. Give it {days:.1f} days to soften.",
            "Still green outside. It's unripe. Wait {days:.1f} days.",
            "Not sweet yet! Status: Unripe. Ripening in {days:.1f} days.",
            "Keep it in a warm spot. This unripe papaya needs {days:.1f} days.",
            "Check it again in {days:.1f} days. Currently unripe.",
            "The flesh is still hard. Unripe for {days:.1f} more days.",
            "Patience for the tropical taste! Unripe for {days:.1f} days.",
            "The model sees a green papaya. Target: {days:.1f} days.",
            "Wait for the yellow skin. Currently unripe for {days:.1f} days.",
            "Not ready for your salad. Status: Unripe. Wait {days:.1f} days.",
            "Bit too firm to eat. It's unripe. Days left: {days:.1f}.",
            "Give it some time to ripen up. Unripe for {days:.1f} days.",
            "Immature fruit detected. Status: Unripe. Time: {days:.1f} days.",
            "Still on its way! This papaya is unripe for {days:.1f} days.",
            "Predicted as unripe. It should be perfect in {days:.1f} days."
        ],
        "fresh": [
            "Soft and sweet! This fresh papaya is good for {days:.1f} days.",
            "Perfect tropical treat! It's fresh and ready for {days:.1f} days.",
            "Look at that orange flesh! Status: Fresh for {days:.1f} days.",
            "Beautifully ripe papaya. Enjoy it within the next {days:.1f} days.",
            "Firm but yielding. It's fresh for another {days:.1f} days.",
            "Ideal ripeness reached! Status: Fresh. Time left: {days:.1f} days.",
            "Ready for slicing. This fresh papaya has {days:.1f} days left.",
            "A healthy breakfast choice! Fresh for {days:.1f} more days.",
            "High quality fruit alert. Fresh for the next {days:.1f} days.",
            "Savor it today! It's fresh for up to {days:.1f} days.",
            "Vibrant and tasty. Status: Fresh. Window: {days:.1f} days.",
            "The model loves this fresh papaya! Best in {days:.1f} days.",
            "Peak sweetness detected! Enjoy for {days:.1f} more days.",
            "Healthy and ready. It's fresh and good for {days:.1f} days.",
            "Top tier papaya specimen! Status: Fresh for {days:.1f} days."
        ],
        "rotten": [
            "This papaya has spoiled. It's rotten and {days:.1f} days past due.",
            "Very mushy and smelly. It's rotten. Expired {days:.1f} days ago.",
            "Spoilage found near the stem. It's rotten for {days:.1f} days.",
            "Not fit for eating. Status: Rotten. Age: {days:.1f} days.",
            "Do not consume! This papaya is rotten and old by {days:.1f} days.",
            "Leaking juices detected. It's rotten. Over: {days:.1f} days.",
            "Nature's cleanup started. Status: Rotten. Past: {days:.1f} days.",
            "The model flags this as a spoiled papaya. Old by {days:.1f} days.",
            "Throw it out; it's been rotten for {days:.1f} days.",
            "The tropical fruit has turned. Status: Rotten. Day {days:.1f}.",
            "Avoid this one; it's rotten. Total days past fresh: {days:.1f}.",
            "Deep decay found. Status: Rotten. Expired {days:.1f} days ago.",
            "The shelf life ended {days:.1f} days ago. It's rotten.",
            "Identified as decomposed. It's been rotten for {days:.1f} days.",
            "It's a goner! This papaya is rotten and {days:.1f} days past juice-able."
        ]
    },
    "unknown": {
        "unripe": ["This item appears to be unripe. Estimated time: {days:.1f} days."],
        "fresh": ["This item looks fresh! Good for another {days:.1f} days."],
        "rotten": ["This item is rotten. It was fresh {days:.1f} days ago."]
    }
}

def load_models():
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
    else:
        print("⚠️ Warning: Identity model not found. Automatic fruit detection disabled.")
        id_model = None

    if id_model:
        id_model.to(DEVICE)
        id_model.eval()

    return class_model, reg_model, id_model

def predict(image_path, manual_fruit="unknown"):
    class_model, reg_model, id_model = load_models()

    # Process image
    image = Image.open(image_path).convert("RGB")
    input_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # 1. Get ripeness stage
        class_output = class_model(input_tensor)
        stage_idx = class_output.argmax(dim=1).item()
        stage = STAGE_LABELS[stage_idx]

        # 2. Get days remaining
        reg_output = reg_model(input_tensor)
        days = reg_output.item()

        # 3. Get fruit identity (if model exists)
        fruit_name = manual_fruit.lower().strip()
        if fruit_name == "unknown" and id_model:
            id_output = id_model(input_tensor)
            fruit_idx = id_output.argmax(dim=1).item()
            fruit_name = FRUIT_LABELS[fruit_idx]
            print(f"🔍 AI detected: {fruit_name.capitalize()}")

    # Fallback to templates
    if fruit_name not in TEMPLATES:
        fruit_name = "unknown"

    # Pick a random template
    template_list = TEMPLATES[fruit_name][stage]
    sentence = random.choice(template_list)
    
    return sentence.format(days=max(0, days) if stage != "rotten" else abs(days))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="🍇 Fully Automatic Fruit Predictor")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("--fruit", default="unknown", help="Optional: Manually specify fruit name")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"❌ Error: File not found at {args.image}")
    else:
        result = predict(args.image, args.fruit)
        print("\n" + "="*60)
        print(f"🤖 AI ANALYSIS: {result}")
        print("="*60 + "\n")

