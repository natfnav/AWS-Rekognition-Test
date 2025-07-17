#!/usr/local/bin/python3

import cv2
import numpy as np
import boto3
import os
import sys
import time
import pyaudio
import contextlib
from botocore.exceptions import BotoCoreError, ClientError

# AWS and Audio Setup
region = 'us-east-2'
reko = boto3.client('rekognition', region_name=region)
polly = boto3.client("polly", region_name=region)
pya = pyaudio.PyAudio()

# Utility for ignoring stderr (used when speaking to suppress ALSA warnings)
@contextlib.contextmanager
def ignore_stderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)

# Polly TTS
def speak(text_string, voice="Joanna"):
    try:
        response = polly.synthesize_speech(
            Text=text_string,
            TextType="text",
            OutputFormat="pcm",
            VoiceId=voice
        )
    except (BotoCoreError, ClientError) as error:
        print(error)
        return

    if "AudioStream" in response:
        stream = pya.open(format=pya.get_format_from_width(width=2),
                          channels=1,
                          rate=16000,
                          output=True)
        stream.write(response['AudioStream'].read())
        time.sleep(0.5)
        stream.stop_stream()
        stream.close()

# Detect labels with Rekognition
def detect_labels_from_frame(frame):
    resized = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
    success, encoded_image = cv2.imencode('.png', resized)
    if not success:
        return []

    response = reko.detect_labels(
        Image={'Bytes': encoded_image.tobytes()},
        MaxLabels=6,
        MinConfidence=60
    )
    return [label['Name'] for label in response['Labels']]

# Main Live Detection Loop
def live_object_detection():
    cam = cv2.VideoCapture(0)
    frame_count = 0
    PROCESS_EVERY_N_FRAMES = 30
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
                    with ignore_stderr():
                        speak("I see: " + label_str)
            except Exception as e:
                print("Error calling Rekognition:", e)

        # Show current label string on screen
        cv2.putText(frame, label_str, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Live Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cam.release()
    cv2.destroyAllWindows()

# Run the main function
if __name__ == "__main__":
    live_object_detection()
