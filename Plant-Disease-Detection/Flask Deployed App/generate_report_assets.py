import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
import CNN
import os
import sys

def generate_dummy_model():
    print("Generating dummy model...", flush=True)
    model = CNN.CNN(34)  # Updated for Pakistan dataset (34 classes)
    # Save the dummy model
    try:
        torch.save(model.state_dict(), "plant_disease_model_1_latest.pt")
        print("Saved plant_disease_model_1_latest.pt", flush=True)
    except Exception as e:
        print(f"Failed to save model: {e}", flush=True)
    return model

def generate_test_image():
    print("Generating test image...")
    # Create a green image with some noise to simulate a leaf
    img = Image.new('RGB', (224, 224), color=(34, 139, 34)) # Forest Green
    draw = ImageDraw.Draw(img)
    # Add some random "lesions"
    for _ in range(5):
        x = np.random.randint(0, 224)
        y = np.random.randint(0, 224)
        r = np.random.randint(10, 30)
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(139, 69, 19)) # Saddle Brown
    
    img.save("test_leaf.jpg")
    print("Saved test_leaf.jpg")
    return img

def generate_gradcam(model, image_path):
    print("Generating Grad-CAM visualization...")
    model.eval()
    
    # Preprocess image
    img = Image.open(image_path)
    img = img.resize((224, 224))
    input_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    input_tensor = input_tensor.unsqueeze(0)
    
    # Hook for gradients
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
        
    def forward_hook(module, input, output):
        activations.append(output)
        
    # Hook the last conv layer
    # Based on CNN.py, the last conv layer is in self.conv_layers
    # The structure is sequential, let's find the last Conv2d
    target_layer = None
    for layer in model.conv_layers:
        if isinstance(layer, nn.Conv2d):
            target_layer = layer
            
    if target_layer:
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)
        
    # Forward pass
    output = model(input_tensor)
    pred_idx = output.argmax(dim=1).item()
    
    # Backward pass
    model.zero_grad()
    output[0, pred_idx].backward()
    
    if not gradients or not activations:
        print("Failed to capture gradients or activations.")
        return

    # Generate heatmap
    grads = gradients[0].cpu().data.numpy()[0]
    fmap = activations[0].cpu().data.numpy()[0]
    
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    
    for i, w in enumerate(weights):
        cam += w * fmap[i]
        
    cam = np.maximum(cam, 0)
    cam = cam / cam.max() if cam.max() > 0 else cam
    
    # Resize heatmap to image size
    # Manual resize using PIL
    heatmap_img = Image.fromarray(cam)
    heatmap_img = heatmap_img.resize((224, 224), resample=Image.BILINEAR)
    heatmap_resized = np.array(heatmap_img)
    
    # Apply colormap (simple red-yellow-blue approximation or just red)
    # Let's do a simple Red overlay
    # Create an RGBA image
    overlay = Image.new('RGBA', (224, 224), (255, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # We can just manipulate pixel data
    # A simple "Jet" like map:
    # Low: Blue, Mid: Green, High: Red
    
    def get_jet_color(val):
        # val is 0 to 1
        # 0.0 -> 0.0, 0.0, 1.0 (Blue)
        # 0.5 -> 0.0, 1.0, 0.0 (Green)
        # 1.0 -> 1.0, 0.0, 0.0 (Red)
        # This is simplified
        r, g, b = 0, 0, 0
        if val < 0.5:
            # Blue to Green
            # 0 -> B=255, G=0
            # 0.5 -> B=0, G=255
            ratio = val * 2
            b = int(255 * (1 - ratio))
            g = int(255 * ratio)
        else:
            # Green to Red
            # 0.5 -> G=255, R=0
            # 1.0 -> G=0, R=255
            ratio = (val - 0.5) * 2
            g = int(255 * (1 - ratio))
            r = int(255 * ratio)
        return (r, g, b, 128) # Alpha 128

    # Create heatmap image
    heatmap_rgba = Image.new('RGBA', (224, 224))
    pixels = heatmap_rgba.load()
    
    for y in range(224):
        for x in range(224):
            val = heatmap_resized[y, x]
            pixels[x, y] = get_jet_color(val)
            
    # Overlay
    img = img.convert("RGBA")
    superimposed_img = Image.alpha_composite(img, heatmap_rgba)
    
    superimposed_img.save("gradcam_output.png")
    print("Saved gradcam_output.png (PIL based)", flush=True)

def main():
    model = generate_dummy_model()
    generate_test_image()
    
    try:
        generate_gradcam(model, "test_leaf.jpg")
    except Exception as e:
        print(f"Grad-CAM generation failed: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
