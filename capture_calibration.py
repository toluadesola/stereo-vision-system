import cv2
import os
import time
from picamera2 import Picamera2
 
# Output folders
os.makedirs("calibration_images/left",  exist_ok=True)
os.makedirs("calibration_images/right", exist_ok=True)
 
# Cameras
camL = Picamera2(0)
camR = Picamera2(1)
 
config = {"size": (640, 480), "format": "RGB888"}
camL.configure(camL.create_preview_configuration(main=config))
camR.configure(camR.create_preview_configuration(main=config))
camL.start()
camR.start()
time.sleep(1)   
 
# Checkerboard settings (must match calibration.py)
CHECKERBOARD = (9, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
count = 0
 
print("Cameras started.")
print("Position the RIGID board against a wall ? do NOT hold it.")
print("Press 's' to save pair (only saved if BOTH cameras see the board).")
print("Press 'q' to quit.")
 
while True:
 
    rawL = camL.capture_array()
    rawR = camR.capture_array()
 
    # Strip alpha if present
    if rawL.ndim == 3 and rawL.shape[2] == 4:
        rawL = rawL[:, :, :3]
    if rawR.ndim == 3 and rawR.shape[2] == 4:
        rawR = rawR[:, :, :3]
 
    # Convert to BGR so cv2.imshow and cv2.imwrite work correctly
    frameL = cv2.cvtColor(rawL, cv2.COLOR_RGB2BGR)
    frameR = cv2.cvtColor(rawR, cv2.COLOR_RGB2BGR)
 
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
 
    # Live checkerboard overlay so you can see what the camera sees
    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)
 
    displayL = frameL.copy()
    displayR = frameR.copy()
 
    if retL:
        cv2.drawChessboardCorners(displayL, CHECKERBOARD, cornersL, retL)
    if retR:
        cv2.drawChessboardCorners(displayR, CHECKERBOARD, cornersR, retR)
 
    # Status bar at the top of each preview
    both_ok = retL and retR
    colour   = (0, 200, 0) if both_ok else (0, 0, 200)
    statusL  = "LEFT  OK" if retL else "LEFT  --"
    statusR  = "RIGHT OK" if retR else "RIGHT --"
    cv2.putText(displayL, f"{statusL}  pairs={count}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    cv2.putText(displayR, f"{statusR}  pairs={count}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
 
    cv2.imshow("LEFT  CAMERA", displayL)
    cv2.imshow("RIGHT CAMERA", displayR)
 
    key = cv2.waitKey(1) & 0xFF
 
    if key == ord('s'):
        if not both_ok:
            # Flash rejection message
            print(f"  Pair rejected ? board not detected in both cameras "
                  f"(L={retL}, R={retR}).  Reposition and try again.")
        else:
            # Refine corners before saving
            cL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            cR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
 
            left_path  = f"calibration_images/left/{count}.jpg"
            right_path = f"calibration_images/right/{count}.jpg"
            cv2.imwrite(left_path,  frameL)
            cv2.imwrite(right_path, frameR)
            print(f"  Saved pair {count}: {left_path}, {right_path}")
            count += 1
 
    elif key == ord('q'):
        break
 
cv2.destroyAllWindows()
camL.stop()
camR.stop()
print(f"\nDone. {count} pairs saved.")
print("Now run:  python calibration.py")
 