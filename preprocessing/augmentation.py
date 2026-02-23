import os
import cv2
import numpy as np
import albumentations as A
import shutil
from pathlib import Path

class ImageAugmenter:
    """
    Augments images with color-based transformations (no geometric changes, NO RESIZING).
    Generates 3x augmented dataset while preserving labels.
    """
    
    def __init__(self, input_images_dir, input_labels_dir, output_dir):
        """
        Args:
            input_images_dir: Path to directory containing original images
            input_labels_dir: Path to directory containing labels (txt files for YOLO format)
            output_dir: Path to save augmented images and labels
        """
        self.input_images_dir = input_images_dir
        self.input_labels_dir = input_labels_dir
        self.output_dir = output_dir
        
        # Create output directories
        self.output_images_dir = os.path.join(output_dir, 'images')
        self.output_labels_dir = os.path.join(output_dir, 'labels')
        
        os.makedirs(self.output_images_dir, exist_ok=True)
        os.makedirs(self.output_labels_dir, exist_ok=True)
        
        # Define AUGMENTATION ONLY pipeline (matching Roboflow config)
        # NO RESIZING - just color-based transformations
        self.transform = A.Compose([
            # Grayscale: Apply to 15% of images
            A.ToGray(p=0.15),
            
            # Hue shift: Between -15° and +15°
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=0, val_shift_limit=0, p=1.0),
            
            # Saturation shift: Between -25% and +25%
            A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=25, val_shift_limit=0, p=1.0),
            
            # Brightness: Between -15% and +15%
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0, p=1.0),
            
            # Exposure: Between -10% and +10%
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0, p=1.0),
            
            # Blur: Up to 2.5px
            A.Blur(blur_limit=2, p=1.0),
            
            # Noise: Up to 1.45% of pixels
            A.ISONoise(p=1.0),
        ])
    
    def augment_image(self, image):
        """Apply augmentation transforms to image"""
        augmented = self.transform(image=image)
        return augmented['image']
    
    def process_dataset(self):
        """Process all images and create 3x augmented dataset"""
        image_files = sorted([f for f in os.listdir(self.input_images_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        
        total_images = len(image_files)
        print(f"Found {total_images} images to process")
        print("-" * 70)
        
        successful = 0
        failed = 0
        
        for idx, image_file in enumerate(image_files):
            try:
                image_path = os.path.join(self.input_images_dir, image_file)
                label_file = os.path.splitext(image_file)[0] + '.txt'
                label_path = os.path.join(self.input_labels_dir, label_file)
                
                # Read image (NO RESIZING)
                original_image = cv2.imread(image_path)
                if original_image is None:
                    print(f"✗ Failed to read: {image_file}")
                    failed += 1
                    continue
                
                # Convert BGR to RGB for albumentations
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                
                # Get base filename without extension
                base_name = os.path.splitext(image_file)[0]
                
                # Save original image (without augmentation)
                output_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(self.output_images_dir, f"{base_name}_0.jpg"), output_image)
                
                # Copy label for original image
                if os.path.exists(label_path):
                    shutil.copy(label_path, os.path.join(self.output_labels_dir, f"{base_name}_0.txt"))
                
                # Create 2 additional augmented versions (total 3x)
                for aug_num in range(1, 3):
                    augmented_image = self.augment_image(original_image)
                    
                    # Convert back to BGR for saving
                    output_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                    output_path = os.path.join(self.output_images_dir, f"{base_name}_{aug_num}.jpg")
                    cv2.imwrite(output_path, output_image)
                    
                    # Copy label for augmented image
                    if os.path.exists(label_path):
                        shutil.copy(label_path, os.path.join(self.output_labels_dir, f"{base_name}_{aug_num}.txt"))
                
                successful += 1
                
                if (idx + 1) % 50 == 0:
                    print(f"Progress: {idx + 1}/{total_images}")
            
            except Exception as e:
                print(f"✗ Error processing {image_file}: {str(e)}")
                failed += 1
        
        print("-" * 70)
        print(f"\n✓ Augmentation complete!")
        print(f"Original images: {total_images}")
        print(f"Total augmented images: {total_images * 3}")
        print(f"Successful: {successful}/{total_images}")
        print(f"Failed: {failed}/{total_images}")
        print(f"Output directory: {self.output_dir}")


def main():
    """
    Main function to run augmentation.
    Modify these paths according to your directory structure.
    """
    
    # Configuration
    INPUT_IMAGES_DIR = "Fold_Datasets/3_tools/SLT/Fold100/images"      # Already resized images
    INPUT_LABELS_DIR = "Fold_Datasets/3_tools/SLT/Fold100/labels"      # Corresponding labels
    OUTPUT_DIR = "Fold_Datasets/Augmented/SLT/Fold100/"                 # Where to save augmented data
    
    # Validate input directories
    if not os.path.exists(INPUT_IMAGES_DIR):
        print(f"Error: Images directory not found: {INPUT_IMAGES_DIR}")
        return
    
    if not os.path.exists(INPUT_LABELS_DIR):
        print(f"Error: Labels directory not found: {INPUT_LABELS_DIR}")
        return
    
    print("\nStarting image augmentation (COLOR-BASED ONLY, NO RESIZING)...")
    print(f"Input images: {INPUT_IMAGES_DIR}")
    print(f"Input labels: {INPUT_LABELS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 70)
    
    # Initialize augmenter and process dataset
    augmenter = ImageAugmenter(
        input_images_dir=INPUT_IMAGES_DIR,
        input_labels_dir=INPUT_LABELS_DIR,
        output_dir=OUTPUT_DIR
    )
    
    augmenter.process_dataset()


if __name__ == "__main__":
    main()