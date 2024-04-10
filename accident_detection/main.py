import cv2
from detection import AccidentDetectionModel
import numpy as np
# import winsound
import os

model = AccidentDetectionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX


cap = cv2.VideoCapture('./file.mp4') # for using on video files
# cap = cv2.VideoCapture(0) # for using with camera
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read() # if frame is read correctly ret is True
    # frame = cv2.flip(frame, 1)
    if not ret:
        print("Can't receive frame...")
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(gray_frame, (250, 250))

    pred, prob = model.predict_accident(roi[np.newaxis, :, :])
    if pred == "Accident":
        prob = round(prob[0][0] * 100, 2)

        # to beep when alert:
        if(prob >= 90):            
            # winsound.Beep(440, 650)
            os.system("beep -f 2000 -l 1500")
            cv2.rectangle(frame, (0, 0), (350, 40), (0, 0, 0), -1)
            cv2.putText(frame, "Accident Happened", (20, 30), font, 1.0, (255, 0 , 255), 2)

        # cv2.putText(frame, pred + " " + str(prob), (20, 30), font, 1, (255, 255, 0), 2) # for printing prediction percentage value
    cv2.imshow('Camera Preview', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()