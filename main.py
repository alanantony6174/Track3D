from ultralytics import YOLO
from tracker import register_tracker

model = YOLO("yolo11n.pt")

# Register the local tracker
register_tracker(model, persist=True)

# Use predict with the registered tracker
model.predict(
    source=0,        # webcam
    show=True,
    classes=[0],
    tracker="botsort.yaml"
)