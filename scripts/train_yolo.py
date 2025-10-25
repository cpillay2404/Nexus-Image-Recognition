import argparse, os
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="exp")
    args = ap.parse_args()

    model = YOLO(args.model)  # pre-trained checkpoint
    results = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, project=args.project, name=args.name)
    print(results)

if __name__ == "__main__":
    main()
