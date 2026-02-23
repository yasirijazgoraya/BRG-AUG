


import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, '..')

# Configuration
CONFIG = {
    'weights': 'baseline_tsne_experiment/yolov5m_baseline/weights/best.pt',
    'synthetic_dir': '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/augmented_train_data/train/images',
    'real_dir': '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/test/images',
    'output_dir': 'gradcam_results',
    'img_size': 640,
    'device': 'cuda:0',
    'n_samples': 6,  # Number of samples per category
}


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, class_idx=None):
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            # Use the max prediction
            if isinstance(output, tuple):
                output = output[0]
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        
        if isinstance(output, tuple):
            output = output[0]
        
        # Create one-hot encoding
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(CONFIG['img_size'], CONFIG['img_size']), 
                           mode='bilinear', align_corners=False)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy()


def apply_colormap(img, cam, alpha=0.5):
    """Overlay CAM on image"""
    cam_colored = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
    
    img_np = np.array(img)
    overlay = (alpha * cam_colored + (1 - alpha) * img_np).astype(np.uint8)
    
    return overlay


def load_model(weights_path, device):
    """Load YOLOv5 model"""
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model = ckpt['model'].float() if 'model' in ckpt else ckpt.float()
    model.to(device)
    model.eval()
    return model


def process_image(img_path, img_size=640):
    """Load and preprocess image"""
    img = Image.open(img_path).convert('RGB')
    img_resized = img.resize((img_size, img_size))
    
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    
    return img_resized, img_tensor


def main():
    import random
    random.seed(42)
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    print("Loading model...")
    model = load_model(CONFIG['weights'], CONFIG['device'])
    
    # Get target layer (SPPF - layer 9)
    target_layer = model.model[9]
    gradcam = GradCAM(model, target_layer)
    
    # Get sample images
    from pathlib import Path
    
    synthetic_images = list(Path(CONFIG['synthetic_dir']).glob('*.jpg'))
    real_images = list(Path(CONFIG['real_dir']).glob('*.jpg'))
    
    # Sample images
    syn_samples = random.sample(synthetic_images, min(CONFIG['n_samples'], len(synthetic_images)))
    real_samples = random.sample(real_images, min(CONFIG['n_samples'], len(real_images)))
    
    # Create comparison figure
    fig, axes = plt.subplots(4, CONFIG['n_samples'], figsize=(3*CONFIG['n_samples'], 12), dpi=150)
    
    print("Generating Grad-CAM visualizations...")
    
    for i, img_path in enumerate(syn_samples):
        img, img_tensor = process_image(str(img_path), CONFIG['img_size'])
        img_tensor = img_tensor.to(CONFIG['device']).requires_grad_(True)
        
        try:
            cam = gradcam.generate(img_tensor)
            overlay = apply_colormap(img, cam)
            
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'Syn {i+1}', fontsize=10)
            axes[0, i].axis('off')
            
            axes[1, i].imshow(overlay)
            axes[1, i].axis('off')
        except Exception as e:
            print(f"Error with synthetic {i}: {e}")
            axes[0, i].text(0.5, 0.5, 'Error', ha='center')
            axes[1, i].text(0.5, 0.5, 'Error', ha='center')
    
    for i, img_path in enumerate(real_samples):
        img, img_tensor = process_image(str(img_path), CONFIG['img_size'])
        img_tensor = img_tensor.to(CONFIG['device']).requires_grad_(True)
        
        try:
            cam = gradcam.generate(img_tensor)
            overlay = apply_colormap(img, cam)
            
            axes[2, i].imshow(img)
            axes[2, i].set_title(f'Real {i+1}', fontsize=10)
            axes[2, i].axis('off')
            
            axes[3, i].imshow(overlay)
            axes[3, i].axis('off')
        except Exception as e:
            print(f"Error with real {i}: {e}")
            axes[2, i].text(0.5, 0.5, 'Error', ha='center')
            axes[3, i].text(0.5, 0.5, 'Error', ha='center')
    
    # Add row labels
    axes[0, 0].set_ylabel('Synthetic\nOriginal', fontsize=12)
    axes[1, 0].set_ylabel('Synthetic\nGrad-CAM', fontsize=12)
    axes[2, 0].set_ylabel('Real\nOriginal', fontsize=12)
    axes[3, 0].set_ylabel('Real\nGrad-CAM', fontsize=12)
    
    plt.suptitle('Grad-CAM: Model Attention on Synthetic vs Real Samples', fontsize=14)
    plt.tight_layout()
    
    output_path = os.path.join(CONFIG['output_dir'], 'gradcam_comparison.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    
    print(f"\nSaved: {output_path}")
    print("Done!")


if __name__ == '__main__':
    main()
