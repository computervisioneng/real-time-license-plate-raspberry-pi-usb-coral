import time

from ultralytics import YOLO
import cv2


model = YOLO('license_plate_detector_int8_edgetpu.tflite', task='detect')

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 10, (frame_width, frame_height))

print('process starts')
ret = True
while ret:
    tic = time.time()
    ret, frame = cap.read()
    if ret:
        output = model(frame, imgsz=320, verbose=False)

        for i, det in enumerate(output[0].boxes.data.tolist()):
            x1, y1, x2, y2, score, class_id = det

            if score < 0.6:
                continue

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3) 
            cv2.putText(frame, 'license_plate', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, 'FPS: ' + str(int(1 / (time.time() - tic))), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

        out.write(frame)


cap.release()
out.release()
cv2.destroyAllWindows()
