##192.168.2.164/mjpg/video.mjpg

import cv2 
cap = cv2.VideoCapture("http://root:admin@85.24.203.44:5432/mjpg/video.mjpg")
key = cv2. waitKey(1)
imgNum = 0
while True:
    try:
        check, frame = cap.read()
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            imgNum = imgNum + 1
            cv2.imwrite(filename='saved_img' + str(imgNum) + '.jpg', img=frame)
            cv2.destroyAllWindows()
    
        elif key == ord('q'):
            print("Turning off camera.")
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
        
    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break
