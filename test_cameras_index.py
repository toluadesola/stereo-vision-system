"""
Camera index identification test.

HOW TO USE:
  1. Run the script: two windows appear side by side
  2. Cover the LEFT physical camera lens with your finger
     -> The window that goes BLACK is that camera's index
  3. Uncover, then cover the RIGHT physical camera lens
     -> The window that goes BLACK is that camera's index

That tells you definitively which Picamera2 index is which.
Press Q to quit.
"""

import cv2
from picamera2 import Picamera2

camL = Picamera2(0)
camR = Picamera2(1)

config = {"size": (640, 480), "format": "BGR888"}
camL.configure(camL.create_preview_configuration(main=config))
camR.configure(camR.create_preview_configuration(main=config))
camL.start()
camR.start()

print("=================================================")
print("Cover your LEFT camera lens with your finger.")
print("Whichever window goes dark = that camera's index.")
print("Then do the same for the RIGHT camera.")
print("Press Q to quit.")
print("=================================================")

while True:
    frameL = camL.capture_array()[:, :, :3]
    frameR = camR.capture_array()[:, :, :3]

    cv2.putText(frameL, "Picamera2(0)", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frameR, "Picamera2(1)", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Picamera2(0)", frameL)
    cv2.imshow("Picamera2(1)", frameR)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camL.stop()
camR.stop()
cv2.destroyAllWindows()