import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys, os

# Loading the model
json_file = open("json_files/model_2.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("json_files/model_2.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

# Category dictionary
categories = {1: 'OKAY', 2: 'PEACE', 3: 'CLENCHED FIST', 4: 'I LOVE YOU', 5: 'THUMB UP' ,6:'unknown',}

while True:
    _, frame = cap.read()
    # Simulating mirror image
    #frame = cv2.flip(frame, 1)

    # Got this from collect-data.py
    # Coordinates of the ROI
    x1 = int(10)
    y1 = 10
    x2 = int(0.5*frame.shape[1])
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    #ROI
    roi = frame[y1:y2, x1:x2]

    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (64, 64))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("test", test_image)
    # Batch of 1
    result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
    # print(result)
    prediction = {'CLENCHED_FIST': result[0][0],
                  'I LOVE YOU': result[0][1],
                  'OKAY': result[0][2],
                  'PEACE': result[0][3],
                  'THUMB UP': result[0][4],
                  'unknown' :result[0][5] ,
                  }
    # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)

    # Displaying the predictions
    # if result==0:
    #     cv2.putText(frame, 'ERROR', (x1, y2+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 1)
    #     cv2.imshow("Frame", frame)
    # else:
    cv2.putText(frame, prediction[0][0], (x1, y2+30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 1)
    cv2.imshow("Frame", frame)


    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break


cap.release()
cv2.destroyAllWindows()
