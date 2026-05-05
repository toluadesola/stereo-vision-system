import threading
from picamera2 import Picamera2


def start_cameras():

    camL = Picamera2(0)
    camR = Picamera2(1)

    config = {
        "size": (640, 480),
        "format": "BGR888"
    }

    camL.configure(camL.create_preview_configuration(main=config))
    camR.configure(camR.create_preview_configuration(main=config))

    camL.start()
    camR.start()

    return camL, camR


def capture_sync(camL, camR):
    """
    Capture both cameras in parallel threads to minimise the time gap
    between the two frames. 
    """
    frames = [None, None]

    def grab(idx, cam):
        frames[idx] = cam.capture_array()

    t0 = threading.Thread(target=grab, args=(0, camL))
    t1 = threading.Thread(target=grab, args=(1, camR))
    t0.start()
    t1.start()
    t0.join()
    t1.join()

    return frames[0], frames[1]