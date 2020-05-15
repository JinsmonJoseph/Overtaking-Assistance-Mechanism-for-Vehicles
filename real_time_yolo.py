import cv2
import numpy as np
import time

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
count=[]
objects=["bicycle","car","motorbike","bus","person","truck"]
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
cap = cv2.VideoCapture("dataset2.mp4")

font = cv2.FONT_HERSHEY_TRIPLEX
font2= cv2.FONT_ITALIC
starting_time = time.time()

frame_id = 0
flag=[]
while True:
    _, frame = cap.read()
    flag.insert(frame_id,0)
    count.append(0)
    
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                distance=round((700-y)/6)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            area=w*h
            confidence = confidences[i]
            #color = colors[class_ids[i]]
            color1=(0,255,0)
            color2=(0,0,255)
            if(label in objects):
                count[frame_id-1]=count[frame_id-1]+1
                if(x>250):
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color2, 2)
                    cv2.rectangle(frame, (x, y-30), (x + 150, y), (0,0,0), -1)
                    cv2.putText(frame, label , (x, y-5), font2,1, color2, 3)
                    cv2.rectangle(frame, (x, y+h), (x + 100, y+h+20), (0,0,0),-1)
                    cv2.putText(frame,"dist-"+str(round((600-(y+h))/10))+"m" , (x, y+h+15), font2, 1, color2, 3)
                    if(flag[frame_id]==0):
                        cv2.rectangle(frame, (0,0), (400,40),(0,0,0),-1)
                        cv2.putText(frame, "DO NOT OVERTAKE ", (10,30), font, 1, color2, 3)
                        flag[frame_id]=1
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color1, 2)
                    cv2.rectangle(frame, (x, y-30), (x + 150, y), (0,0,0),-1)
                    cv2.putText(frame, label , (x, y-5), font2, 1, color1, 3)
                    cv2.rectangle(frame, (x, y+h), (x + 100, y+h+20), (0,0,0),-1)
                    cv2.putText(frame,"dist-"+str(round((600-(y+h))/10))+"m" , (x, y+h+15), font2, 1, color1, 3)
            else:
               continue

    if(flag[frame_id]==0):
        cv2.rectangle(frame, (0,0), (400,40),(0,0,0),-1)
        cv2.putText(frame, "SAFE FOR OVERTAKING" , (10,30), font, 1, color1, 3)
    cv2.rectangle(frame, (0,660), (270,700),(0,0,0),-1)
    cv2.putText(frame, "Frame No.- " + str(frame_id), (10,690), font,1, (255,255,255), 3)
    cv2.rectangle(frame, (500,660), (720,700),(0,0,0),-1)
    cv2.putText(frame,"objects - "+ str(count[frame_id-1]), (520,685), font,1, (255,255,255), 3)
    frame_id += 1
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    
    cv2.imshow("image", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()