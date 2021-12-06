import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle, argparse 
from datetime import datetime


cap = cv2.VideoCapture(0)

if not cap.isOpened():                                       # Check if the webcam is opened correctly
    raise IOError(" Webcam can not open !!! ")

while True:
    _, frame = cap.read()

    # Load model YOLOv3
    MODEL = 'yolo/yolov3-face.cfg'
    WEIGHT = 'yolo/yolov3-wider_16000.weights'
    IMG_WIDTH, IMG_HEIGHT = 416, 416 
    
    net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT) 
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Detect objects
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(IMG_WIDTH, IMG_HEIGHT), mean=[0, 0, 0], swapRB=1, crop=False)
    net.setInput(blob)                                      
    output_layers = net.getUnconnectedOutLayersNames()      
    outs = net.forward(output_layers)                    

    # Detect bounding boxes with high confidence
    frame_height    = frame.shape[0]
    frame_width     = frame.shape[1]
            
    confidences = []
    boxes = []
    final_boxes = []

    for out in outs:                                         
        for detection in out:                                
            confidence = detection[-1]
            if confidence > 0.5:                            
                center_x    = int(detection[0] * frame_width)
                center_y    = int(detection[1] * frame_height)
                width       = int(detection[2] * frame_width)  
                height      = int(detection[3] * frame_height)
                
                topleft_x   = int(center_x - width/2)       
                topleft_y   = int(center_y - height/2)
                confidences.append(float(confidence))
                boxes.append(([topleft_x, topleft_y, width, height]))
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)   

    # Drawing bounding box
    result = frame.copy()

    for i in indices:
        box = boxes[i]
        final_boxes.append(box)
                
        left    = box[0]            
        top     = box[1]
        width   = box[2]
        height  = box[3]
        
        cv2.rectangle(result, (left, top), (left + width + 5, top + height + 5), (0,0,255), 2)         
        text = f'{confidences[i]:.2f}'

        cv2.putText(result, text, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (1, 190, 200), 2)  
        face_quantity = f'Detected face(s): {len(final_boxes)}'    

        cv2.putText(result, face_quantity, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    # Display the result
    cv2.imshow('Smart camera:', result)
    
    c = cv2.waitKey(1)       # Press 'ESC' to exit capturing
    if c == 27:                 
        break

    elif  c == ord("s"):    # Press 'S' to stop capturing
        try: 
            for box in final_boxes:
                crop = frame[(top - 35):(top + height + 35),(left - 35):(left + width + 35)] 
                image = cv2.resize(crop, (300, 300), interpolation = cv2.INTER_AREA)
                image = image / 255.
                image = image.flatten()
                now = datetime.now().strftime("%H%M%S")             
                cv2.imwrite(f'media/{now}.jpg',crop)                  
        except Exception as ex:
              print(ex)

# Release the capture
cap.release()
cv2.destroyAllWindows()