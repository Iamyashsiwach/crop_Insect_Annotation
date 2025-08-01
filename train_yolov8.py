#!/usr/bin/env python3
"""
YOLOv8 Training Script for Crop Insect Detection
"""

from ultralytics import YOLO
import os

def train_detection_model():
    """Train YOLOv8 detection model"""
    print("🚀 Starting YOLOv8 Detection Training...")
    
    # Load a model
    model = YOLO('yolov8s.pt')  # load a pretrained model
    
    # Train the model
    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='crop_insect_detection',
        patience=20,  # early stopping
        save=True,
        plots=True
    )
    
    print("✅ Detection training completed!")
    return results

def train_segmentation_model():
    """Train YOLOv8 segmentation model"""
    print("🚀 Starting YOLOv8 Segmentation Training...")
    
    # Load a model
    model = YOLO('yolov8s-seg.pt')  # load a pretrained segmentation model
    
    # Train the model
    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='crop_insect_segmentation',
        patience=20,  # early stopping
        save=True,
        plots=True
    )
    
    print("✅ Segmentation training completed!")
    return results

def quick_test():
    """Quick test with minimal epochs to verify setup"""
    print("🧪 Running quick test...")
    
    model = YOLO('yolov8n.pt')  # use nano model for quick test
    
    results = model.train(
        data='data.yaml',
        epochs=1,
        imgsz=640,
        batch=8,
        name='crop_insect_test'
    )
    
    print("✅ Quick test completed!")
    return results

if __name__ == "__main__":
    print("YOLOv8 Crop Insect Training")
    print("=" * 40)
    
    # Check if data.yaml exists
    if not os.path.exists('data.yaml'):
        print("❌ Error: data.yaml not found!")
        exit(1)
    
    print("Select training mode:")
    print("1. Quick test (1 epoch)")
    print("2. Detection training (100 epochs)")
    print("3. Segmentation training (100 epochs)")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    try:
        if choice == "1":
            quick_test()
        elif choice == "2":
            train_detection_model()
        elif choice == "3":
            train_segmentation_model()
        else:
            print("❌ Invalid choice!")
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("💡 Tip: Try reducing batch size if you get memory errors") 