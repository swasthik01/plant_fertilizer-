"""
Training script for Soil Detection Model
Train EfficientNet model on soil images
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from models.soil_detection.soil_detector import (
    SoilDetectionModel,
    SoilDetectorTrainer,
    SoilImageDataset,
    get_data_transforms
)
import os
import glob


def prepare_dataset(data_dir, train_split=0.8):
    """
    Prepare training and validation datasets
    Expected directory structure:
    data_dir/
        Sandy/
            image1.jpg
            image2.jpg
        Loamy/
            image1.jpg
        ...
    """
    soil_types = ['Sandy', 'Loamy', 'Clayey', 'Silty', 'Peaty', 'Chalky']
    
    all_images = []
    all_labels = []
    
    for idx, soil_type in enumerate(soil_types):
        soil_dir = os.path.join(data_dir, soil_type)
        if os.path.exists(soil_dir):
            images = glob.glob(os.path.join(soil_dir, '*.jpg'))
            images += glob.glob(os.path.join(soil_dir, '*.png'))
            
            all_images.extend(images)
            all_labels.extend([idx] * len(images))
    
    # Split into train and validation
    from sklearn.model_selection import train_test_split
    
    train_images, val_images, train_labels, val_labels = train_test_split(
        all_images, all_labels, train_size=train_split, random_state=42, stratify=all_labels
    )
    
    return train_images, val_images, train_labels, val_labels


def train_model(data_dir, epochs=50, batch_size=32, learning_rate=0.001):
    """Train the soil detection model"""
    
    print("Preparing dataset...")
    train_images, val_images, train_labels, val_labels = prepare_dataset(data_dir)
    
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    
    # Get transforms
    train_transform, val_transform = get_data_transforms()
    
    # Create datasets
    train_dataset = SoilImageDataset(train_images, train_labels, train_transform)
    val_dataset = SoilImageDataset(val_images, val_labels, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model
    print("Initializing model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = SoilDetectionModel(num_classes=6, pretrained=True)
    
    # Create trainer
    trainer = SoilDetectorTrainer(model, device)
    
    # Train
    print("Starting training...")
    trained_model = trainer.train(
        train_loader, 
        val_loader, 
        epochs=epochs, 
        lr=learning_rate,
        save_path='models/soil_detection/best_model.pth'
    )
    
    # Plot training history
    trainer.plot_training_history('models/soil_detection/training_history.png')
    
    print("Training completed!")
    return trained_model


if __name__ == "__main__":
    # Set data directory
    data_dir = "data/soil_images"
    
    # Check if data exists
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found.")
        print("Please organize your soil images in the following structure:")
        print("data/soil_images/")
        print("  Sandy/")
        print("  Loamy/")
        print("  Clayey/")
        print("  Silty/")
        print("  Peaty/")
        print("  Chalky/")
    else:
        # Train model
        model = train_model(
            data_dir=data_dir,
            epochs=50,
            batch_size=32,
            learning_rate=0.001
        )
        
        print("Model saved to: models/soil_detection/best_model.pth")
