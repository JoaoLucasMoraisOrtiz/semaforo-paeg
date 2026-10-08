import argparse
import os
from ultralytics import RTDETR

def main():
    parser = argparse.ArgumentParser(description='Export RT-DETR Model for Edge inference')
    parser.add_argument('--format', type=str, default='onnx', choices=['onnx', 'openvino'],
                        help='Format to export (onnx or openvino)')
    parser.add_argument('--weights', type=str, default='rtdetr-l.pt',
                        help='Path to the model weights')
    parser.add_argument('--int8', action='store_true', help='Use INT8 Quantization (great for OpenVINO/Intel CPUs)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image resolution to export (smaller is faster)')
    
    args = parser.parse_args()
    
    model_path = args.weights
    if not os.path.exists(model_path) and model_path == 'runs/detect/train/weights/best.pt':
        print(f"⚠️ Model {model_path} not found. Defaulting to rtdetr-l.pt")
        model_path = 'rtdetr-l.pt'
        
    print(f"🔄 Loading model: {model_path}")
    model = RTDETR(model_path)
    
    print(f"⚙️ Exporting model to {args.format} format (imgsz={args.imgsz}, INT8={args.int8})...")
    exported_path = model.export(format=args.format, simplify=True, half=False, int8=args.int8, imgsz=args.imgsz)
    
    print(f"✅ Export successful! Optimized model saved at: {exported_path}")

if __name__ == '__main__':
    main()
