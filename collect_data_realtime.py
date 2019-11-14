import cv2
import numpy as np
import os

# Create the directory structure
if not os.path.exists("data"):
    os.makedirs("data")
    os.makedirs("data/train")
    os.makedirs("data/train/okay")
    os.makedirs("data/train/peace")
    os.makedirs("data/train/Clenched_fist")
    os.makedirs("data/train/ILY")
    os.makedirs("data/train/Thumb_up")
    os.makedirs("data/test")
    os.makedirs("data/test/okay")
    os.makedirs("data/test/peace")
    os.makedirs("data/test/Clenched_fist")
    os.makedirs("data/test/ILY")
    os.makedirs("data/test/Thumb_up")



# Train or test
mode = 'train'
directory = 'data/'+mode+'/'

cap = cv2.VideoCapture(0)

print("Press 1 to save OKAY")
print("Press 2 to save PEACE")
print("Press 3 to save CLENCHED_FIST")
print("Press 4 to save ILY")
print("Press 5 to save for THUMB_UP")

while True:
    _, frame = cap.read()
    # Simulating mirror image
    #frame = cv2.flip(frame, 1)

    # Getting count of existing images
    count = {'okay': len(os.listdir(directory+"/okay")),
             'peace': len(os.listdir(directory+"/peace")),
             'Clenched_fist': len(os.listdir(directory+"/Clenched_fist")),
             'ILY': len(os.listdir(directory+"/ILY")),
             'Thumb_up': len(os.listdir(directory+"/Thumb_up")),
             }

    # Printing the count in each set to the screen
    cv2.putText(frame, "MODE : "+mode, (int(0.5*frame.shape[1])+20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(frame, "IMAGE COUNT", (int(0.5*frame.shape[1])+20, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(frame, "OKAY : "+str(count['okay']), (int(0.5*frame.shape[1])+20, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(frame, "PEACE : "+str(count['peace']), (int(0.5*frame.shape[1])+20, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(frame, "Clenched Fist : "+str(count['Clenched_fist']), (int(0.5*frame.shape[1])+20, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(frame, "ILY : "+str(count['ILY']), (int(0.5*frame.shape[1])+20, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    cv2.putText(frame, "Thumb Up : "+str(count['Thumb_up']), (int(0.5*frame.shape[1])+20, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)

    # Coordinates of the ROI (Region Of Interest)
    x1 = int(10)
    y1 = 10
    x2 = int(0.5*frame.shape[1])
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    #ROI
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (64, 64))

    cv2.imshow("Frame", frame)

    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("ROI", roi)

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # escape key
        break
    if interrupt & 0xFF == ord('1'):
        cv2.imwrite(directory+'okay/'+str(count['okay'])+'.jpg', roi)
    if interrupt & 0xFF == ord('2'):
        cv2.imwrite(directory+'peace/'+str(count['peace'])+'.jpg', roi)
    if interrupt & 0xFF == ord('3'):
        cv2.imwrite(directory+'Clenched_fist/'+str(count['Clenched_fist'])+'.jpg', roi)
    if interrupt & 0xFF == ord('4'):
        cv2.imwrite(directory+'ILY/'+str(count['ILY'])+'.jpg', roi)
    if interrupt & 0xFF == ord('5'):
        cv2.imwrite(directory+'Thumb_up/'+str(count['Thumb_up'])+'.jpg', roi)

cap.release()
cv2.destroyAllWindows()
