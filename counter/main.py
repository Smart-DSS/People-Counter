import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *
import cvzone
import json
import time
from datetime import datetime
import pytz

# Initialize YOLO model
model = YOLO('yolov8s.pt')

# Mouse event callback function for debugging
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

# Open video capture
cap = cv2.VideoCapture(r"C:\Users\steph\Desktop\crowd analysis\peoplecounter\A07_20240605134952-----.mp4")

# Read class list from coco.txt
with open("coco.txt", "r") as my_file:
    data = my_file.read()
class_list = data.split("\n")

# Initialize tracking variables
count = 0
persondown = {}
tracker = Tracker()
counter1 = []
personup = {}
counter2 = []
cy1 = 333
cy2 = 298
offset = 6

# Initialize total count
total = 0

# List to store JSON data for each frame
json_list = []
exceed_count_data = []

# Initialize interval and storage variables
hourly_interval_duration = 3600  # Interval duration for hourly data in seconds (1 hour = 3600 seconds)
exceed_interval_duration = 60    # Interval duration for exceed count data in seconds (1 minute = 60 seconds)
store_interval_duration = 60     # Interval duration for storing data in seconds (1 minute = 60 seconds)

# Initialize last storage times
last_hourly_store_time = time.time()
last_exceed_store_time = time.time()
last_store_time = time.time()

# Define IST timezone
ist = pytz.timezone('Asia/Kolkata')

# Initial storage flag
initial_storage_done = False

# Create window and set mouse callback
cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

# Main processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue
    
    # Resize frame for processing
    frame = cv2.resize(frame, (1020, 500))

    # Predict with YOLO model
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list = []

    # Filter 'person' class detections
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        
        c = class_list[d]
        if 'person' in c:
            list.append([x1, y1, x2, y2])
        
    # Update tracker with detected person bounding boxes
    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        cv2.circle(frame, (cx, cy), 4, (155, 0, 255), -1)
        if cy1 < (cy + offset) and cy1 > (cy - offset):
            cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
            persondown[id] = (cx, cy)
        if id in persondown:
            if cy2 < (cy + offset) and cy2 > (cy - offset):
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                if counter1.count(id) == 0:
                    counter1.append(id)

        if cy2 < (cy + offset) and cy2 > (cy - offset):
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
            personup[id] = (cx, cy)
        if id in personup:
            if cy1 < (cy + offset) and cy1 > (cy - offset):
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                if counter2.count(id) == 0:
                    counter2.append(id)

    # Draw counting lines
    cv2.line(frame, (37, cy1), (845, cy1), (255, 0, 0), 2, 2)
    cv2.line(frame, (127, cy2), (825, cy2), (0, 255, 0), 2, 2)
   
    # Calculate counts (Exit, Enter, Total)
    Exit = len(counter1)
    Enter = len(counter2)
    total = Exit - Enter 

    # Prepare JSON data for current frame
    timestamp_ist = datetime.now(ist)  # Get the current time in IST

    data_dict = {
        "timestamp_ist": timestamp_ist.strftime('%Y-%m-%d %H:%M:%S'),  # Format IST timestamp
        "Exit": Exit,
        "Enter": Enter,
        "Total": total
    }

    # Store initial total count data immediately
    if not initial_storage_done:
        json_list.append(data_dict)
        with open("hourly_data.json", "a") as json_file:
            json.dump(json_list, json_file, indent=4)
        json_list = []  # Reset JSON list after initial storage
        initial_storage_done = True
        last_hourly_store_time = time.time()  # Update last hourly storage time to current time

    # Check if the storage interval has passed
    current_time = time.time()
    if current_time - last_store_time >= store_interval_duration:
        # Append JSON data to list for hourly data
        json_list.append(data_dict)
        last_store_time = current_time  # Update last storage time with current time

    # Check if an hour has passed since last hourly storage
    if current_time - last_hourly_store_time >= hourly_interval_duration:
        # Read existing hourly data
        try:
            with open("hourly_data.json", "r") as json_file:
                existing_data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = []

        # Append new data to existing data
        existing_data.extend(json_list)

        # Store the updated hourly data
        with open("hourly_data.json", "w") as json_file:
            json.dump(existing_data, json_file, indent=4)
        json_list = []  # Reset JSON list
        last_hourly_store_time = current_time  # Update last hourly storage time with current time

    # Check if total count exceeds 3 and store exceed count data once per minute
    if total > 3:
        if current_time - last_exceed_store_time >= exceed_interval_duration:  # Check if one minute has passed
            exceed_count_data.append(data_dict)
            with open("exceed_count_data.json", "a") as json_file:
                json.dump(exceed_count_data, json_file, indent=4)
            last_exceed_store_time = current_time  # Update last exceed data storage time with current time
            print("People count exceeded")
            cvzone.putTextRect(frame, "Warning: People count exceeded!", (50, 100), scale=2, thickness=2, colorR=(0, 0, 255), offset=10)
       
    # Display counts on frame
    cvzone.putTextRect(frame, f'Exit: {Exit}', (50, 60), 1, 2)
    cvzone.putTextRect(frame, f'Enter: {Enter}', (50, 40), 1, 2)
    cvzone.putTextRect(frame, f'Total: {total}', (50, 80), 1, 2)

    # Display frame
    cv2.imshow("RGB", frame)
    
    # Check for key press (Esc key to exit)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
