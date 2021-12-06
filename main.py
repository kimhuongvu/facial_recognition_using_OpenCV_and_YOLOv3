import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle, argparse 
from datetime import datetime
import tensorflow as tf

#load model, labels for prediction 
IMG_SIZE = 224
Model_Path = 'Face_Recognition_model.h5'
my_model = tf.keras.models.load_model(Model_Path)
class_names = ['Cuong', 'Ha', 'Huong', 'Huyen', 'Jenny', 'Quyen']       # changing class_names according to training model


#named for camera window
camera_window = 'WHO ARE THEY?'
cv2.namedWindow(camera_window, cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(0) #webcam 

if not cap.isOpened():                                      
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()

    YOLO_MODEL = 'yolo/yolov3-face.cfg'
    YOLO_WEIGHT = 'yolo/yolov3-wider_16000.weights'
    net = cv2.dnn.readNetFromDarknet(YOLO_MODEL, YOLO_WEIGHT) 
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # detect image frame by Blobbing
    IMG_WIDTH, IMG_HEIGHT = 416, 416  
    CONF_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4

    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(IMG_WIDTH, IMG_HEIGHT), mean=[0, 0, 0], swapRB=1, crop=False)
    net.setInput(blob)                                      
    output_layers = net.getUnconnectedOutLayersNames()      
    outs = net.forward(output_layers) 

    # Detect bounding boxes with high confidence
    frame_height    = frame.shape[0]    
    frame_width     = frame.shape[1]
    
    confidences = []
    boxes       = []
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

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    for i in indices:
        box = boxes[i]
        confidence = confidences[i]
        final_boxes.append([box, confidence])

    if final_boxes != 0:
        for box, confidence in final_boxes:
            left    = box[0]
            top     = box[1]
            width   = box[2]
            height  = box[3]

            try:    
                cropped     = frame[(top - 35 ):(top + height + 35),(left - 35):(left + width + 35)] 
                # pre-processing image for prediction:
                img         = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
                img_array   = np.expand_dims(img, axis=0)
                prediction = my_model.predict([img_array])
                index = np.argmax(prediction.flatten())                 
                name = class_names[index]
            except:
                name = 'Detecting...'
                print('Unrecognizable!')

            # Drawing bounding box & display label (class_names) on frame
            cv2.rectangle(frame, (left, top), (left + width + 5, top + height + 5), (0,0,255), 2)
            label_display = f'{name} - {confidence:.2f}'
            cv2.putText(frame, label_display, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2) 
                        
        # Display number of face detected                        
        number_of_faces = f'Detected face(s): {len(final_boxes)}'
        cv2.putText(frame, number_of_faces, (30, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,0,0), 2)

        
    cv2.imshow(camera_window, frame)

    c = cv2.waitKey(1)
    if c == 27:    
        break

cap.release()
cv2.destroyAllWindows()