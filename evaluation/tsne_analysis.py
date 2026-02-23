#!/usr/bin/env python3
"""
t-SNE Feature Analysis for BRG-AUG Paper
Uses verified baseline model to extract and visualize features
"""

import os
import sys
import random
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for YOLOv5 imports
sys.path.insert(0, '..')

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ============== CONFIGURATION ==============
CONFIG = {
    # Paths (using verified baseline)
    'weights': 'baseline_tsne_experiment/yolov5m_baseline/weights/best.pt',
    'synthetic_dir': '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/augmented_train_data/train/images',
    'real_dir': '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/test/images',
    'output_dir': 'tsne_results',
    
    # Sampling (adjust based on memory)
    'n_synthetic_samples': 500,
    'n_real_samples': 500,  # Max 540 available
    
    # Image processing
    'img_size': 640,
    
    # t-SNE parameters
    'perplexity': 30,
    'n_iter': 1000,
    
    # Device
    'device': 'cuda:0'
}


class FeatureExtractor:
    """Extract features from YOLOv5 backbone"""
    
    def __init__(self, weights_path, device='cuda:0'):
        self.device = device
        self.features = []
        
        print(f"Loading model from {weights_path}...")
        
        # Load checkpoint
        #self.ckpt = torch.load(weights_path, map_location=device)
        self.ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        
        if 'model' in self.ckpt:
            self.model = self.ckpt['model'].float()
        else:
            self.model = self.ckpt.float()
        
        self.model.to(device)
        self.model.eval()
        
        # Register hook
        self._register_hook()
        
        print(f"Model loaded on {device}")
    
    def _register_hook(self):
        """Register forward hook on SPPF layer (layer 9)"""
        target_layer = self.model.model[9]
        
        def hook_fn(module, input, output):
            # Global Average Pooling
            if len(output.shape) == 4:
                pooled = output.mean(dim=[2, 3])
            else:
                pooled = output
            self.features.append(pooled.detach().cpu().numpy())
        
        self.hook = target_layer.register_forward_hook(hook_fn)
    
    def preprocess(self, img_path, img_size=640):
        """Preprocess image"""
        img = Image.open(img_path).convert('RGB')
        img = img.resize((img_size, img_size))
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        return img_tensor.to(self.device)
    
    def extract(self, image_paths, img_size=640):
        """Extract features from images"""
        self.features = []
        
        for img_path in tqdm(image_paths, desc="Extracting features"):
            try:
                img_tensor = self.preprocess(img_path, img_size)
                with torch.no_grad():
                    _ = self.model(img_tensor)
            except Exception as e:
                print(f"Error: {img_path}: {e}")
                continue
        
        return np.vstack(self.features) if self.features else np.array([])
    
    def cleanup(self):
        if hasattr(self, 'hook'):
            self.hook.remove()


def get_image_paths(directory, n_samples=None):
    """Get and sample image paths"""
    images = list(Path(directory).glob('*.jpg'))
    if not images:
        images = list(Path(directory).glob('*.png'))
    
    print(f"Found {len(images)} images in {directory}")
    
    if n_samples and n_samples < len(images):
        images = random.sample(images, n_samples)
        print(f"Sampled {n_samples} images")
    
    return [str(img) for img in images]


def get_class_label(filename, is_synthetic=True):
    """Get class from filename"""
    basename = os.path.basename(filename).lower()
    
    if is_synthetic:
        # Synthetic: *_abnormal*, *_geo_*, *_photo_*, *_misalign*
        if any(x in basename for x in ['abnormal', 'geo_', 'photo_', 'misalign']):
            return 'Abnormal'
        return 'Normal'
    else:
        # Real: HRI_N_* (Normal), HRI_O_* (Abnormal)
        if '_n_' in basename or basename.startswith('hri_n'):
            return 'Normal'
        return 'Abnormal'


def run_tsne(features, perplexity=30, n_iter=1000):
    """Run t-SNE"""
    print(f"Running t-SNE (perplexity={perplexity}, iterations={n_iter})...")
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=42,
        init='pca',
        learning_rate='auto'
    )
    
    embeddings = tsne.fit_transform(features_scaled)
    print("t-SNE completed!")
    
    return embeddings


def plot_domain_tsne(embeddings, labels, output_path):
    """Plot synthetic vs real"""
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    labels = np.array(labels)
    colors = {'Synthetic': '#2E86AB', 'Real': '#E94F37'}
    markers = {'Synthetic': 'o', 'Real': '^'}
    
    for label in ['Synthetic', 'Real']:
        mask = labels == label
        ax.scatter(
            embeddings[mask, 0], embeddings[mask, 1],
            c=colors[label], marker=markers[label],
            label=f'{label} (n={mask.sum()})',
            alpha=0.6, s=50, edgecolors='white', linewidths=0.5
        )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14)
    ax.set_title('Feature Space: Synthetic vs Real Samples', fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_class_tsne(embeddings, domain_labels, class_labels, output_path):
    """Plot by domain and class"""
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    domain_labels = np.array(domain_labels)
    class_labels = np.array(class_labels)
    
    # Colors for class, markers for domain
    colors = {'Normal': '#2E86AB', 'Abnormal': '#E94F37'}
    markers = {'Synthetic': 'o', 'Real': '^'}
    
    for domain in ['Synthetic', 'Real']:
        for cls in ['Normal', 'Abnormal']:
            mask = (domain_labels == domain) & (class_labels == cls)
            if mask.sum() > 0:
                ax.scatter(
                    embeddings[mask, 0], embeddings[mask, 1],
                    c=colors[cls], marker=markers[domain],
                    label=f'{domain} - {cls} (n={mask.sum()})',
                    alpha=0.6, s=50, edgecolors='white', linewidths=0.5
                )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14)
    ax.set_title('Feature Space: Domain and Class Distribution', fontsize=14)
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("t-SNE FEATURE ANALYSIS FOR BRG-AUG")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Verify paths
    if not os.path.exists(CONFIG['weights']):
        print(f"ERROR: Weights not found: {CONFIG['weights']}")
        return
    
    # Initialize extractor
    extractor = FeatureExtractor(CONFIG['weights'], CONFIG['device'])
    
    # Get image paths
    print("\n[1/4] Loading image paths...")
    synthetic_paths = get_image_paths(CONFIG['synthetic_dir'], CONFIG['n_synthetic_samples'])
    real_paths = get_image_paths(CONFIG['real_dir'], CONFIG['n_real_samples'])
    
    # Extract features
    print("\n[2/4] Extracting synthetic features...")
    synthetic_features = extractor.extract(synthetic_paths, CONFIG['img_size'])
    
    print("\n[3/4] Extracting real features...")
    extractor.features = []  # Reset
    real_features = extractor.extract(real_paths, CONFIG['img_size'])
    
    extractor.cleanup()
    
    print(f"\nSynthetic features: {synthetic_features.shape}")
    print(f"Real features: {real_features.shape}")
    
    # Combine
    all_features = np.vstack([synthetic_features, real_features])
    domain_labels = ['Synthetic'] * len(synthetic_features) + ['Real'] * len(real_features)
    
    # Get class labels
    syn_classes = [get_class_label(p, True) for p in synthetic_paths[:len(synthetic_features)]]
    real_classes = [get_class_label(p, False) for p in real_paths[:len(real_features)]]
    class_labels = syn_classes + real_classes
    
    # Print class distribution
    print(f"\nClass distribution:")
    print(f"  Synthetic Normal: {syn_classes.count('Normal')}")
    print(f"  Synthetic Abnormal: {syn_classes.count('Abnormal')}")
    print(f"  Real Normal: {real_classes.count('Normal')}")
    print(f"  Real Abnormal: {real_classes.count('Abnormal')}")
    
    # Run t-SNE
    print("\n[4/4] Running t-SNE...")
    embeddings = run_tsne(all_features, CONFIG['perplexity'], CONFIG['n_iter'])
    
    # Create plots
    print("\nCreating visualizations...")
    
    plot_domain_tsne(
        embeddings, domain_labels,
        os.path.join(CONFIG['output_dir'], 'tsne_synthetic_vs_real.pdf')
    )
    
    plot_class_tsne(
        embeddings, domain_labels, class_labels,
        os.path.join(CONFIG['output_dir'], 'tsne_domain_and_class.pdf')
    )
    
    # Save data
    np.savez(
        os.path.join(CONFIG['output_dir'], 'tsne_embeddings.npz'),
        embeddings=embeddings,
        domain_labels=np.array(domain_labels),
        class_labels=np.array(class_labels)
    )
    
    print("\n" + "=" * 60)
    print("COMPLETED!")
    print("=" * 60)
    print(f"\nResults in: {CONFIG['output_dir']}/")
    print("  - tsne_synthetic_vs_real.pdf  (for paper)")
    print("  - tsne_domain_and_class.pdf   (detailed view)")
    print("  - tsne_embeddings.npz         (raw data)")


if __name__ == '__main__':
    main()
