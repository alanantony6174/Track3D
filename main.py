from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.track(
    source=0,        # webcam
    show=True,
    classes=[0],
    persist=True,
    tracker="botsort.yaml"
)