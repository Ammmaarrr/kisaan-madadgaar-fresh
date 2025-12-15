import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
import pandas as pd
import json
import timm

# Load class names from trained model
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

num_classes = len(class_names)
print(f"✅ Loaded {num_classes} classes")

# Load EfficientNet-B4 model (NEW trained model)
print("🔄 Loading EfficientNet-B4 model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=num_classes)

# Load trained weights
checkpoint = torch.load("pakistan_model_best.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
print(f"✅ Model loaded! ({num_classes} classes)")
print(f"🖥️ Device: {device}")

# Image transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def prediction(image_path):
    image = Image.open(image_path).convert('RGB')
    input_data = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_data)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    index = predicted.item()
    conf = confidence.item() * 100
    
    print(f"🔍 Prediction: {class_names[index]} ({conf:.1f}%)")
    return index

# Try to load disease info (may need to update for new classes)
try:
    disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
    supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')
    print("✅ Disease info loaded")
except Exception as e:
    print(f"⚠️ Disease info not found, using class names directly")
    # Create simple disease info from class names
    disease_info = pd.DataFrame({
        'disease_name': class_names,
        'description': [f"Disease detected: {name}" for name in class_names],
        'Possible Steps': ["Please consult an agricultural expert for treatment options." for _ in class_names],
        'image_url': ["" for _ in class_names]
    })
    supplement_info = pd.DataFrame({
        'supplement name': ["Consult Expert" for _ in class_names],
        'supplement image': ["" for _ in class_names],
        'buy link': ["" for _ in class_names]
    })

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(f"📷 Image saved: {file_path}")
        
        pred = prediction(file_path)
        
        # Get disease info (handle index out of range)
        if pred < len(disease_info):
            title = disease_info['disease_name'][pred]
            description = disease_info['description'][pred]
            prevent = disease_info['Possible Steps'][pred]
            image_url = disease_info['image_url'][pred]
        else:
            title = class_names[pred]
            description = f"Detected: {class_names[pred]}"
            prevent = "Please consult an agricultural expert."
            image_url = ""
        
        if pred < len(supplement_info):
            supplement_name = supplement_info['supplement name'][pred]
            supplement_image_url = supplement_info['supplement image'][pred]
            supplement_buy_link = supplement_info['buy link'][pred]
        else:
            supplement_name = "Consult Expert"
            supplement_image_url = ""
            supplement_buy_link = ""
        
        return render_template('submit.html', title=title, desc=description, prevent=prevent,
                             image_url=image_url, pred=pred, sname=supplement_name,
                             simage=supplement_image_url, buy_link=supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', 
                          supplement_image=list(supplement_info['supplement image']),
                          supplement_name=list(supplement_info['supplement name']),
                          disease=list(disease_info['disease_name']),
                          buy=list(supplement_info['buy link']))

@app.route('/classes')
def show_classes():
    """Show all disease classes the model can detect"""
    return render_template('classes.html', classes=class_names, count=len(class_names))

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 کسان مددگار - Plant Disease Detection")
    print("="*50)
    print(f"📊 Model: EfficientNet-B4")
    print(f"🎯 Classes: {num_classes}")
    print(f"🌐 Open: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True)
