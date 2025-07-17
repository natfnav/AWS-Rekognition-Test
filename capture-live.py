#!/usr/local/bin/python3

import cv2
import boto3

# AWS Region and Rekognition Setup
region = 'us-east-2'
reko = boto3.client('rekognition', region_name=region)

# Detect labels with Rekognition
def detect_labels_from_frame(frame):
    resized = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
    success, encoded_image = cv2.imencode('.png', resized)
    if not success:
        return []

    response = reko.detect_labels(
        Image={'Bytes': encoded_image.tobytes()},
        MaxLabels=6,
        MinConfidence=90
    )
    return [f"{label['Name']}: {label['Confidence']:.2f}%" for label in response['Labels']]

# Main Live Detection Loop
def live_object_detection():
    cam = cv2.VideoCapture(0)
    frame_count = 0
    PROCESS_EVERY_N_FRAMES = 60
    last_labels = []
    label_str = ""

    print("Starting live object detection. Press ESC to quit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            try:
                new_labels = detect_labels_from_frame(frame)
                if new_labels != last_labels:
                    label_str = ", ".join(new_labels)
                    last_labels = new_labels
                    print("Detected:", label_str)
            except Exception as e:
                print("Error calling Rekognition:", e)

        # Show current label string on screen
        y_offset = 30
        for label in last_labels:
            cv2.putText(frame, label, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30  # Move to next line

        cv2.imshow("Live Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cam.release()
    cv2.destroyAllWindows()

# Run the main function
if __name__ == "__main__":
    live_object_detection()
