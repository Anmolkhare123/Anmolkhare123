# run.py

import cv2
from model import load_model, EuclideanDistTracker, show_image
from IPython.display import display, clear_output, Image

# Video Capture
cap = cv2.VideoCapture(r'H:\workflow\video2.mp4')
tracker = EuclideanDistTracker()
model = load_model()

# Video Writer setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
output_video_path = 'output_video.avi'
fps = 30  # Frames per second
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Initialize counters and sets to track unique object IDs
car_count = 0
bike_count = 0
animal_count = 0
unique_car_ids = set()
unique_bike_ids = set()
unique_animal_ids = set()

# Define labels to count
vehicle_labels = ['car', 'motorcycle', 'bicycle']
animal_labels = ['dog', 'cat', 'cow']

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)

    detections = []
    detected_labels = []

    for result in results:
        for detection in result.boxes:
            x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy().astype(int)
            conf = detection.conf[0].item()
            cls = detection.cls[0].item()
            label = model.names[int(cls)]

            if label in vehicle_labels + animal_labels and conf > 0.4:
                detections.append([x1, y1, x2 - x1, y2 - y1])
                detected_labels.append(label)

    # Object tracking
    boxes_ids = tracker.update(detections)

    # Draw detection and tracking results
    for idx, box_id in enumerate(boxes_ids):
        x, y, w, h, object_id = box_id
        label = detected_labels[idx] if idx < len(detected_labels) else "Unknown"
        cv2.putText(frame, f'{label} ID: {object_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Update counts and track unique IDs
        if label == 'car' and object_id not in unique_car_ids:
            car_count += 1
            unique_car_ids.add(object_id)
        elif label in ['motorcycle', 'bicycle'] and object_id not in unique_bike_ids:
            bike_count += 1
            unique_bike_ids.add(object_id)
        elif label in animal_labels and object_id not in unique_animal_ids:
            animal_count += 1
            unique_animal_ids.add(object_id)

    # Display counts on the frame
    cv2.putText(frame, f'Cars: {car_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Bikes: {bike_count}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Animals: {animal_count}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write the frame to the output video file
    out.write(frame)

    # Display the frame
    clear_output(wait=True)  # Clear previous output
    show_image(frame)  # Display the frame with detections and counts

# Release resources
cap.release()
out.release()  # Release the VideoWriter
