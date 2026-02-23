import os
import cv2
import numpy as np
from pathlib import Path


class ImageResizer:
    """
    Resizes images and adjusts YOLO format labels accordingly.
    Applies letterboxing (padding with black borders) to maintain aspect ratio.
    Saves resized images and adjusted labels to output directory.
    """
    
    def __init__(self, input_images_dir, input_labels_dir, output_dir, target_size=640):
        """
        Args:
            input_images_dir: Path to directory containing original images
            input_labels_dir: Path to directory containing YOLO labels (txt files)
            output_dir: Path to save resized images and adjusted labels
            target_size: Target image size (default 640x640)
        """
        self.input_images_dir = input_images_dir
        self.input_labels_dir = input_labels_dir
        self.output_dir = output_dir
        self.target_size = target_size
        
        # Create output directories
        self.output_images_dir = os.path.join(output_dir, 'images')
        self.output_labels_dir = os.path.join(output_dir, 'labels')
        
        os.makedirs(self.output_images_dir, exist_ok=True)
        os.makedirs(self.output_labels_dir, exist_ok=True)
    
    def resize_image_with_padding(self, image):
        """
        Resize image to target size with letterboxing (black padding).
        Returns: resized image, scale factor, padding info
        """
        h, w = image.shape[:2]
        
        # Calculate scale to fit image in target_size
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create canvas with black background
        canvas = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        
        # Calculate padding
        y_offset = (self.target_size - new_h) // 2
        x_offset = (self.target_size - new_w) // 2
        
        # Place resized image on canvas
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas, scale, x_offset, y_offset
    
    def adjust_yolo_labels(self, labels_text, scale, x_offset, y_offset, original_h, original_w):
        """
        Adjust YOLO format labels for resized image.
        
        YOLO format: <class_id> <x_center> <y_center> <width> <height>
        All values are normalized (0-1) relative to image dimensions.
        
        Args:
            labels_text: Original label text
            scale: Scale factor applied to image
            x_offset: Horizontal padding offset
            y_offset: Vertical padding offset
            original_h: Original image height
            original_w: Original image width
        """
        if not labels_text.strip():
            return ""
        
        adjusted_labels = []
        lines = labels_text.strip().split('\n')
        
        for line in lines:
            if not line.strip():
                continue
            
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = parts[0]
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Convert normalized coordinates to pixel coordinates (original image)
            x_center_px = x_center * original_w
            y_center_px = y_center * original_h
            width_px = width * original_w
            height_px = height * original_h
            
            # Apply scale
            x_center_px_scaled = x_center_px * scale
            y_center_px_scaled = y_center_px * scale
            width_px_scaled = width_px * scale
            height_px_scaled = height_px * scale
            
            # Apply padding offsets
            x_center_px_final = x_center_px_scaled + x_offset
            y_center_px_final = y_center_px_scaled + y_offset
            
            # Convert back to normalized coordinates (new image size: target_size x target_size)
            x_center_norm = x_center_px_final / self.target_size
            y_center_norm = y_center_px_final / self.target_size
            width_norm = width_px_scaled / self.target_size
            height_norm = height_px_scaled / self.target_size
            
            # Clamp values to [0, 1] range
            x_center_norm = max(0, min(1, x_center_norm))
            y_center_norm = max(0, min(1, y_center_norm))
            width_norm = max(0, min(1, width_norm))
            height_norm = max(0, min(1, height_norm))
            
            adjusted_labels.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")
        
        return '\n'.join(adjusted_labels)
    
    def process_dataset(self):
        """Process all images and resize with label adjustment"""
        image_files = sorted([f for f in os.listdir(self.input_images_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        
        total_images = len(image_files)
        print(f"Found {total_images} images to process")
        print(f"Target size: {self.target_size}x{self.target_size}")
        print("-" * 70)
        
        successful = 0
        failed = 0
        skipped_no_labels = 0
        
        for idx, image_file in enumerate(image_files):
            try:
                image_path = os.path.join(self.input_images_dir, image_file)
                label_file = os.path.splitext(image_file)[0] + '.txt'
                label_path = os.path.join(self.input_labels_dir, label_file)
                
                # Read image
                original_image = cv2.imread(image_path)
                if original_image is None:
                    print(f"✗ Failed to read: {image_file}")
                    failed += 1
                    continue
                
                # Get original image dimensions
                original_h, original_w = original_image.shape[:2]
                
                # Resize image with padding
                resized_image, scale, x_offset, y_offset = self.resize_image_with_padding(original_image)
                
                # Save resized image
                output_image_path = os.path.join(self.output_images_dir, image_file)
                cv2.imwrite(output_image_path, resized_image)
                
                # Process labels if they exist
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        labels_text = f.read()
                    
                    # Adjust labels for resized image (passing original dimensions)
                    adjusted_labels = self.adjust_yolo_labels(labels_text, scale, x_offset, y_offset, original_h, original_w)
                    
                    # Save adjusted labels
                    output_label_path = os.path.join(self.output_labels_dir, label_file)
                    with open(output_label_path, 'w') as f:
                        f.write(adjusted_labels)
                    
                    successful += 1
                else:
                    # Save image but no labels
                    skipped_no_labels += 1
                    print(f"⚠ No label file for: {image_file}")
                
                if (idx + 1) % 50 == 0:
                    print(f"Progress: {idx + 1}/{total_images}")
            
            except Exception as e:
                print(f"✗ Error processing {image_file}: {str(e)}")
                failed += 1
        
        print("-" * 70)
        print(f"\n✓ Resizing complete!")
        print(f"Successful: {successful}/{total_images}")
        print(f"Skipped (no labels): {skipped_no_labels}/{total_images}")
        print(f"Failed: {failed}/{total_images}")
        print(f"\nOutput:")
        print(f"  Images: {self.output_images_dir}")
        print(f"  Labels: {self.output_labels_dir}")


def main():
    """
    Main function to run image resizing.
    Modify these paths according to your directory structure.
    """
    
    # Configuration
    INPUT_IMAGES_DIR = "Fold_Datasets/3_tools_ready_data/Test/images"      # Directory with original images
    INPUT_LABELS_DIR = "Fold_Datasets/3_tools_ready_data/Test/labels"      # Directory with YOLO label files
    OUTPUT_DIR = "Fold_Datasets/3_tools_ready_data/Resized_ready_experimentation/Test"                    # Where to save resized images and labels
    TARGET_SIZE = 640                                                # Image size (640x640)
    
    # Validate input directories
    if not os.path.exists(INPUT_IMAGES_DIR):
        print(f"Error: Images directory not found: {INPUT_IMAGES_DIR}")
        return
    
    if not os.path.exists(INPUT_LABELS_DIR):
        print(f"Error: Labels directory not found: {INPUT_LABELS_DIR}")
        return
    
    print("Starting image resizing...")
    print(f"Input images: {INPUT_IMAGES_DIR}")
    print(f"Input labels: {INPUT_LABELS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Target size: {TARGET_SIZE}x{TARGET_SIZE}")
    print("-" * 70)
    
    # Initialize resizer and process dataset
    resizer = ImageResizer(
        input_images_dir=INPUT_IMAGES_DIR,
        input_labels_dir=INPUT_LABELS_DIR,
        output_dir=OUTPUT_DIR,
        target_size=TARGET_SIZE
    )
    
    resizer.process_dataset()


if __name__ == "__main__":
    main()