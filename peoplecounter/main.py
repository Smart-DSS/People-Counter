import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *
import cvzone
import json

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap = cv2.VideoCapture(r"C:\Users\steph\Downloads\pedestrian.mp4")

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0
persondown = {}
tracker = Tracker()
counter1 = []

personup = {}
counter2 = []
cy1 = 144
cy2 = 208
offset = 6

# List to store JSON data for each frame
json_list = []
exceed_count_data = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        
        c = class_list[d]
        if 'person' in c:
            list.append([x1, y1, x2, y2])
        
    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
        if cy1 < (cy + offset) and cy1 > (cy - offset):
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
            persondown[id] = (cx, cy)
        if id in persondown:
            if cy2 < (cy + offset) and cy2 > (cy - offset):
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                if counter1.count(id) == 0:
                    counter1.append(id)

        if cy2 < (cy + offset) and cy2 > (cy - offset):
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
            personup[id] = (cx, cy)
        if id in personup:
            if cy1 < (cy + offset) and cy1 > (cy - offset):
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                if counter2.count(id) == 0:
                    counter2.append(id)

    cv2.line(frame, (319, cy1), (840, cy1), (0, 255, 0), 2, 2)
    cv2.line(frame, (202, cy2), (805, cy2), (0, 255, 255), 2, 2)
   
    Exit = len(counter1)
    Enter = len(counter2)
    total = Exit + Enter 

    data_dict = {
        "Exit": Exit,
        "Enter": Enter,
        "total": total
    }

    json_data = json.dumps(data_dict)
    json_list.append(json_data)

    # Check if total count exceeds 3 and add to the exceed_count_data
    if total > 3:
        exceed_count_data.append(data_dict)
        print("people count exceeded")
        cvzone.putTextRect(frame, "Warning: People count exceeded!", (50, 100), scale=2, thickness=2, colorR=(0, 0, 255), offset=10)
       
              

    cvzone.putTextRect(frame, f'Exit: {Exit}', (50, 60), 1, 2)
    cvzone.putTextRect(frame, f'Enter: {Enter}', (50, 40), 1, 2)
    cvzone.putTextRect(frame, f'Total: {total}', (50, 80), 1, 2)

    # Display JSON data on the frame (optional)
    #cvzone.putTextRect(frame, f'JSON: {json_data}', (50, 100), 1, 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Write the JSON data to a file after the loop
with open("output_data.json", "w") as json_file:
    json.dump(json_list, json_file, indent=4)

with open("exceed_count_data.json", "w") as json_file:
    json.dump(exceed_count_data, json_file, indent=4)


    
