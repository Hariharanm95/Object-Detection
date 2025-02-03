import cv2
from detector import ObjectDetector
from utils_local import plot_boxes

def real_time_detection(detector):
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Object detection
        results = detector.detect_objects(frame)
        frame = plot_boxes(results, frame, detector.model)

        # Display the frame
        cv2.imshow("YOLOv5 Real-Time Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def video_file_detection(video_path, detector):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Object detection
        results = detector.detect_objects(frame)
        frame = plot_boxes(results, frame, detector.model)

        # Display the frame
        cv2.imshow("YOLOv5 Video File Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ObjectDetector()
    
    print("Choose the mode of operation:")
    print("1. Real-Time Detection (Webcam)")
    print("2. Video File Detection")
    choice = input("Enter your choice (1/2): ")

    if choice == "1":
        real_time_detection(detector)
    elif choice == "2":
        video_file = input("Enter the path to the video file: ")
        video_file_detection(video_file, detector)
    else:
        print("Invalid choice. Exiting.")