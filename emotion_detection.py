import os
import cv2
import numpy as np
from fer import FER
import argparse
import sys


def main(video_path):
    # Print the absolute path for debugging
    absolute_path = os.path.abspath(video_path)
    print(f"Attempting to open video file at: {absolute_path}")

    # Initialize the FER emotion detector
    detector = FER()

    # Load OpenCV's pre-trained face detection model (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        print("Please ensure the file exists and the path is correct.")
        return

    # Get video properties for display purposes
    fps = cap.get(cv2.CAP_PROP_FPS)
    width_original  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_original = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Properties:\n - Original Resolution: {width_original}x{height_original}\n - FPS: {fps}")

    # Define desired width for display to fit the screen (e.g., 800 pixels)
    desired_width = 800
    scaling_factor = desired_width / width_original
    desired_height = int(height_original * scaling_factor)

    print(f"Resizing frames to: {desired_width}x{desired_height}")

    # Optional: Define frame skipping for performance (e.g., process every 2nd frame)
    frame_skip = 1  # Set to 1 to process every frame, 2 to process every other frame, etc.
    frame_count = 0

    # Create a resizable window
    cv2.namedWindow('Emotion Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Emotion Detection', desired_width, desired_height)

    while True:
        success, frame = cap.read()
        if not success:
            print("End of video stream or cannot read the frame.")
            break

        frame_count += 1
        if frame_skip > 1 and frame_count % frame_skip != 0:
            # Skip processing this frame
            cv2.imshow('Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Interrupted by user.")
                break
            continue

        # Resize the frame for faster processing
        frame_resized = cv2.resize(frame, (desired_width, desired_height))
        
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )  # Detect faces in the grayscale image

        # Detect emotions in the frame
        emotions = detector.detect_emotions(frame_resized)

        # For each detected face, draw a rectangle and annotate the emotion
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Initialize variables to store the top emotion and its score
            top_emotion = "N/A"
            top_score = 0

            # Find the corresponding face in the emotions list
            for emotion in emotions:
                (ex, ey, ew, eh) = emotion["box"]
                # Simple proximity check
                if abs(x - ex) < 20 and abs(y - ey) < 20:
                    # Get the top emotion
                    emotions_scores = emotion["emotions"]
                    top_emotion, top_score = max(emotions_scores.items(), key=lambda item: item[1])
                    break

            # Annotate the detected emotion on the frame
            cv2.putText(
                frame_resized, f'{top_emotion} ({top_score:.2f})', (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )

        # Display the frame with annotations
        cv2.imshow('Emotion Detection', frame_resized)

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Interrupted by user.")
            break

    # Release the video capture resource and close display windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimized Emotion Detection on Video Files')
    parser.add_argument('video_path', type=str, help='Path to the video file')

    args = parser.parse_args()
    main(args.video_path)
