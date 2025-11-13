"""
CNN-based Soil Type Detection Model using EfficientNet
Model 1: Image-based soil detection
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
from PIL import Image
import os
from typing import Tuple, Dict, List
import json
from pathlib import Path
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class SoilImageDataset(Dataset):
    """Custom Dataset for soil images"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class SoilDetectionModel(nn.Module):
    """EfficientNet-based soil type detection model"""
    
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super(SoilDetectionModel, self).__init__()
        
        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Get the number of features from the last layer
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class ResNetSoilDetector(nn.Module):
    """Alternative ResNet-based soil detector"""
    
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super(ResNetSoilDetector, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Get number of features
        num_features = self.backbone.fc.in_features
        
        # Replace final layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class SoilDetectorTrainer:
    """Training pipeline for soil detection model"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001, save_path='best_model.pth'):
        """Full training loop"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=5, verbose=True)
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            scheduler.step(val_loss)
            
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print('-' * 60)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, save_path)
                print(f'Best model saved with validation accuracy: {val_acc:.2f}%')
        
        return self.model
    
    def plot_training_history(self, save_path='training_history.png'):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Acc')
        ax2.plot(self.val_accuracies, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


class SoilDetectorInference:
    """Inference pipeline for soil detection"""
    
    def __init__(self, model_path: str, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = SoilDetectionModel(num_classes=6)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Soil type mapping
        self.soil_types = {
            0: 'Sandy',
            1: 'Loamy',
            2: 'Clayey',
            3: 'Silty',
            4: 'Peaty',
            5: 'Chalky',
        }
    
    def predict(self, image_path: str) -> Dict:
        """Predict soil type from image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = predicted.item()
        confidence_score = confidence.item()
        
        # Get all class probabilities
        all_probs = {self.soil_types[i]: probabilities[0][i].item() 
                    for i in range(len(self.soil_types))}
        
        # Extract additional features
        features = self.extract_soil_features(image_path)
        
        result = {
            'soil_type': self.soil_types[predicted_class],
            'confidence': confidence_score,
            'all_probabilities': all_probs,
            'color_features': features['color'],
            'texture_features': features['texture'],
            'quality_rating': self.calculate_quality_rating(features),
        }
        
        return result
    
    def extract_soil_features(self, image_path: str) -> Dict:
        """Extract color and texture features from soil image"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Color features
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        color_features = {
            'mean_r': float(np.mean(image_rgb[:, :, 0])),
            'mean_g': float(np.mean(image_rgb[:, :, 1])),
            'mean_b': float(np.mean(image_rgb[:, :, 2])),
            'mean_hue': float(np.mean(hsv[:, :, 0])),
            'mean_saturation': float(np.mean(hsv[:, :, 1])),
            'mean_value': float(np.mean(hsv[:, :, 2])),
        }
        
        # Texture features
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        texture_features = {
            'variance': float(np.var(gray)),
            'roughness': float(np.mean(np.abs(laplacian))),
            'texture_score': float(np.std(gray)),
        }
        
        return {'color': color_features, 'texture': texture_features}
    
    def calculate_quality_rating(self, features: Dict) -> str:
        """Calculate soil quality rating based on features"""
        # Simple heuristic - can be improved
        texture_score = features['texture']['variance']
        
        if texture_score > 1000:
            return 'Good'
        elif texture_score > 500:
            return 'Moderate'
        else:
            return 'Poor'
    
    def batch_predict(self, image_paths: List[str]) -> List[Dict]:
        """Predict soil type for multiple images"""
        results = []
        for img_path in image_paths:
            result = self.predict(img_path)
            results.append(result)
        return results


def get_data_transforms():
    """Get data augmentation transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def evaluate_model(model, test_loader, device, class_names):
    """Evaluate model and generate classification report"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return all_preds, all_labels


if __name__ == "__main__":
    # Example usage
    print("Soil Detection Model initialized")
    print("Available models: EfficientNet-B0, ResNet50")
    
    # Create model
    model = SoilDetectionModel(num_classes=6, pretrained=True)
    print(f"\nModel architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
