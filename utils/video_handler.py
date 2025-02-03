import cv2

def open_video(video_path):
    """Opens the specified video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None
    return cap

def close_video(cap):
    """Releases the video capture and closes all OpenCV windows."""
    if cap:
      cap.release()
    cv2.destroyAllWindows()