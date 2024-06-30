from ultralytics import YOLO
import cv2
import sqlite3

connection = sqlite3.connect("data.db")
cursor = connection.cursor()

sql_query = "SELECT ozellik FROM aracOzellikleri"

cursor.execute(sql_query)

labels = [row[0] for row in cursor.fetchall()]

print(labels)
#%%

model = YOLO("yolov8l.pt")

#%%

warning_msg = "Kavsakta 50'den fazla arac var"
pos_msg = "Kavsakta 50'den az arac var"
change_route_msg = "Yan kavsak bos, oraya gecin"


cam = cv2.VideoCapture(0)

def count_vehicles(frame, region):
    
    x, y, w, h = region
    cropped_frame = frame[y:y+h, x:x+w]

    frame_resize = cv2.resize(cropped_frame, (640, 480))
    rgb_img = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2RGB)
    results = model(rgb_img, verbose=True)
    labels = results[0].names

    vehicle_count = 0
    
    for i in range(len(results[0].boxes)):
        x1, y1, x2, y2 = results[0].boxes.xyxy[i]
        score = results[0].boxes.conf[i]
        label = results[0].boxes.cls[i]
        x1, y1, x2, y2, score, label = int(x1), int(y1), int(x2), int(y2), float(score), int(label)
        name = labels[label]
        
        
        if name in ['person', 'bus', 'car', 'motorcycle', 'truck'] and score > 0.5:  
            vehicle_count += 1
            
            
            cv2.rectangle(cropped_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(cropped_frame, f"{name} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    return vehicle_count, cropped_frame


main_region = (0, 0, 320, 480)  
side_region = (320, 0, 320, 480)  

cv2.namedWindow("Kavşak Kontrolü", cv2.WINDOW_NORMAL)  

while True:
    ret, frame = cam.read()
    
    if not ret:
        break
    
    
    main_car_count, main_frame = count_vehicles(frame, main_region)
    side_car_count, side_frame = count_vehicles(frame, side_region)
    
    # Ana kavşaktaki araç sayısı kontrolü
    if main_car_count >= 2:
        cv2.putText(main_frame, warning_msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        
        if side_car_count < 10:  
            cv2.putText(main_frame, change_route_msg, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    else:
        cv2.putText(main_frame, pos_msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
    cv2.putText(main_frame, "Ana Kavşak Araç Sayısı: " + str(main_car_count), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(side_frame, "Yan Kavşak Araç Sayısı: " + str(side_car_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    
    combined_frame = cv2.hconcat([main_frame, side_frame])

    cv2.imshow("Kavşak Kontrolü", combined_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

connection.close()
