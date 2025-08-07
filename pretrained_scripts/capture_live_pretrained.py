import cv2
import boto3
import random

# AWS Region and Rekognition Setup
region = 'us-east-2'
reko = boto3.client('rekognition', region_name=region)


# Assign unique colors for labels
def get_label_color(label):
    random.seed(hash(label) % (2 ** 32))
    return tuple(random.randint(0, 255) for _ in range(3))


# Compute Intersection over Union (IoU) between two boxes
def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0


# Deduplicate overlapping boxes by keeping highest confidence per region
def deduplicate_labels(labels_info, iou_threshold=0.5):
    labels_info.sort(key=lambda x: float(x['label'].split(':')[1][:-1]), reverse=True)
    filtered = []
    for candidate in labels_info:
        keep = True
        for accepted in filtered:
            if compute_iou(candidate['box'], accepted['box']) > iou_threshold:
                keep = False
                break
        if keep:
            filtered.append(candidate)
    return filtered


# Detect labels with Rekognition
def detect_labels_from_frame(frame):
    resized = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
    success, encoded_image = cv2.imencode('.png', resized)
    if not success:
        return []

    response = reko.detect_labels(
        Image={'Bytes': encoded_image.tobytes()},
        MaxLabels=20,
        MinConfidence=60
    )

    labels_info = []
    resized_h, resized_w = resized.shape[:2]
    orig_h, orig_w = frame.shape[:2]
    scale_x = orig_w / resized_w
    scale_y = orig_h / resized_h

    for label in response['Labels']:
        name = label['Name']
        confidence = label['Confidence']
        color = get_label_color(name)
        if 'Instances' in label:
            for instance in label['Instances']:
                if 'BoundingBox' in instance:
                    box = instance['BoundingBox']
                    left = int(box['Left'] * resized_w * scale_x)
                    top = int(box['Top'] * resized_h * scale_y)
                    width = int(box['Width'] * resized_w * scale_x)
                    height = int(box['Height'] * resized_h * scale_y)
                    labels_info.append({
                        'label': f"{name}: {confidence:.2f}%",
                        'box': (left, top, width, height),
                        'color': color
                    })

    return deduplicate_labels(labels_info)


# Main Live Detection Loop
def live_object_detection():
    cam = cv2.VideoCapture(0)
    frame_count = 0
    PROCESS_EVERY_N_FRAMES = 60
    last_labels = []

    print("Starting live object detection. Press ESC to quit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            try:
                last_labels = detect_labels_from_frame(frame)
                if last_labels:
                    print("Detected:", ", ".join([lbl['label'] for lbl in last_labels]))
            except Exception as e:
                print("Error calling Rekognition:", e)

        # Draw bounding boxes and labels
        for item in last_labels:
            x, y, w_box, h_box = item['box']
            color = item['color']
            label = item['label']
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, item['color'], 2)

        cv2.namedWindow("Live Object Detection", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Live Object Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Live Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cam.release()
    cv2.destroyAllWindows()
