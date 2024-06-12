import numpy as np
import cv2
import time

# Function to perform non-maximum suppression
def non_max_suppression(boxes, confidences, threshold):
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=threshold)
    return indices

# Function to detect vehicles in a frame using YOLO
def detect_vehicles(frame, net, output_layers, classes):
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []
    conf_threshold = 0.5
    vehicle_classes = {1, 2, 3, 5, 7}  # Bicycle, Car, Motorbike, Bus, Truck
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and class_id in vehicle_classes:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indices = non_max_suppression(boxes, confidences, threshold=0.4)
    vehicles = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            vehicles.append((x, y, x + w, y + h, class_ids[i]))
    return vehicles

# Function to estimate distances based on bounding box sizes and camera parameters
def estimate_distances(vehicles, frame_shape, focal_length, real_vehicle_widths, camera_height, camera_angle):
    distances = []
    for vehicle in vehicles:
        x1, y1, x2, y2, class_id = vehicle
        vehicle_width_in_pixels = x2 - x1
        if vehicle_width_in_pixels > 0:  # Ensure width is non-zero to avoid division by zero
            real_width = real_vehicle_widths[class_id]
            distance = (real_width * focal_length) / vehicle_width_in_pixels
            # Adjust for camera height and angle
            adjusted_distance = distance * np.cos(np.radians(camera_angle)) + camera_height * np.sin(np.radians(camera_angle))
            distances.append(adjusted_distance)
        else:
            distances.append(float('inf'))  # Handle case where width is zero (shouldn't happen ideally)
    return distances

# Function to draw LiDAR beams on the frame
def draw_lidar_on_frame(frame, vehicles, distances, user_position, classes):
    close_threshold = 10.0  # Threshold distance to consider a vehicle as passing too close (in meters)

    for vehicle, distance in zip(vehicles, distances):
        x1, y1, x2, y2, class_id = vehicle
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Determine color based on proximity to user
        if distance < close_threshold:
            color = (0, 0, 255)  # Red for vehicles passing too close
        else:
            color = (0, 255, 0)  # Green for other vehicles
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{classes[class_id]}: {distance:.2f} meters', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.line(frame, user_position, (center_x, center_y), (0, 0, 255), 2)
    return frame

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

# Define output layer names directly
output_layers = ['yolo_82', 'yolo_94', 'yolo_106']

# Load video
cap = cv2.VideoCapture("D:\\Headgear\\5927704-hd_1920_1080_30fps.mp4")

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define user positions (assuming positions on the sidewalk)
user_positions = {
    "A": (50, frame_height - 50),
    "B": (frame_width // 4, frame_height - 50),
    "C": (frame_width // 2, frame_height - 50),
    "D": (3 * frame_width // 4, frame_height - 50)
}

# Ask user for their position
user_input = input("Enter your position (A, B, C, D): ").strip().upper()
if user_input in user_positions:
    user_position = user_positions[user_input]
else:
    print("Invalid position entered. Using default position A.")
    user_position = user_positions["A"]

# Define the codec and create VideoWriter object for output
output_path = 'output_with_lidar10.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Time interval for updating distance values
time_interval = 1/6
last_update_time = time.time()

# Camera calibration parameters (assumed or measured)
focal_length = 615  # Assumed focal length of the camera
real_vehicle_widths = {
    1: 0.5,  # Average width of a bicycle in meters
    2: 2.0,  # Average width of a car in meters
    3: 0.7,  # Average width of a motorbike in meters
    5: 2.5,  # Average width of a bus in meters
    7: 2.5,  # Average width of a truck in meters
}
camera_height = 5  # Height of the camera from the ground in meters
camera_angle = 0.0  # Angle of the camera in degrees

last_processed_vehicles = []
last_processed_distances = []

# Main loop to process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Update the bounding boxes in real time
    last_processed_vehicles = detect_vehicles(frame, net, output_layers, classes)

    current_time = time.time()
    # Update the distance values at the specified time interval
    if current_time - last_update_time >= time_interval:
        last_processed_distances = estimate_distances(last_processed_vehicles, frame.shape, focal_length, real_vehicle_widths, camera_height, camera_angle)
        last_update_time = current_time

    # Draw LiDAR beams and display distance information on the frame
    frame_with_lidar = draw_lidar_on_frame(frame.copy(), last_processed_vehicles, last_processed_distances, user_position, classes)
    
    # Write the frame with LiDAR to the output
    out.write(frame_with_lidar)
    
    # Display the frame with LiDAR
    cv2.imshow('Frame with LiDAR', frame_with_lidar)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()