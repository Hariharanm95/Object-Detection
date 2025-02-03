import cv2
from detection import detector, plotter
from utils import video_handler

def main():
    print("Press 'q' to exit the real-time detection window.")
    source_type = input("Choose detection source ('webcam' or 'video'): ").lower()

    if source_type == 'webcam':
        cap = video_handler.open_video(0)  # 0 for default webcam
    elif source_type == 'video':
        video_file = input("Enter the video file path: ")
        cap = video_handler.open_video(video_file)
    else:
        print("Invalid choice.")
        return

    if not cap:
        return

    model = detector.load_model()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.detect_objects(model, frame)
        frame = plotter.plot_boxes(model, results, frame)

        cv2.imshow("YOLOv5 Real-Time Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_handler.close_video(cap)


if __name__ == "__main__":
    main()