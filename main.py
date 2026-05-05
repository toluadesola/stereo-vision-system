import cv2
import numpy as np
from camera import start_cameras, capture_sync
from detector import YOLODetector
from stereo import StereoDepth
from distance import get_object_distance, reset_ema
from alerts import notify
from config import *
 
camL, camR = start_cameras()
 
detector_objects = YOLODetector("models/yolov5n.onnx",        "models/coco.names")
detector_stairs  = YOLODetector("models/yolov5n_stairs.onnx", "models/stairs.names")
stereo           = StereoDepth("calibration/stereo_calibration.npz")
 
MAX_FRAME_COUNT = 210   # LCM(3,5,7) so all skip cycles align cleanly
frame_count     = 0
 
last_objects   = []
last_stairs    = []
last_depth_map = None
disparity = None
 
# Track which labels were seen last cycle so we can reset EMA when
# an object disappears (prevents stale distance from a previous object)
prev_labels: set[str] = set()
 
try:
    while True:
 
        frame_count = (frame_count % MAX_FRAME_COUNT) + 1
 
        # Capture both cameras in parallel to minimise the time gap between the two frames 
        frameL, frameR = capture_sync(camL, camR)
 
        if frameL.ndim == 3 and frameL.shape[2] == 4:
            frameL = frameL[:, :, :3]
        if frameR.ndim == 3 and frameR.shape[2] == 4:
            frameR = frameR[:, :, :3]
 
        # Depth (WLS SGBM most expensive, least frequent)
        if frame_count % DEPTH_EVERY_N == 0:
            try:
                last_depth_map, disparity = stereo.get_depth(
                    frameL, frameR, return_disparity=True
                )
            except Exception as e:
                print(f"[stereo] error: {e}")
                last_depth_map = None
        # Show disparity
        if disparity is not None and disparity.size > 0:
            disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
            disp_vis = disp_vis.astype("uint8")
            cv2.imshow("Disparity Map", disp_vis)
 
        # ===========================
        # SHOW DEPTH MAP
        # ===========================
        if last_depth_map is not None:
 
            depth_vis = last_depth_map.copy()
 
            # Replace NaNs with 0 for display
            depth_vis = np.nan_to_num(depth_vis, nan=0.0)
 
            # Normalize to 0?255
            depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX)
 
            depth_vis = depth_vis.astype("uint8")
 
            # Optional: apply color map (MUCH better)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
 
            cv2.imshow("Depth Map", depth_color)
 
        # YOLO objects 
        if frame_count % DETECT_EVERY_N == 0:
            last_objects = detector_objects.detect(frameL)
 
        # YOLO stairs (staggered so it never runs same frame as objects)
        if frame_count % STAIRS_EVERY_N == 0:
            last_stairs = detector_stairs.detect(frameL)
 
        overlay = frameL.copy()
 
        # Reset EMA for labels that have disappeared
        current_labels = {o["label"] for o in last_objects}
        current_labels.update("stairs" for _ in last_stairs)
        for gone in prev_labels - current_labels:
            reset_ema(gone)
        prev_labels = current_labels
 
        # Draw objects
        objects_with_dist = []
        objects_no_dist   = []
 
        for obj in last_objects:
            entry = dict(obj)
 
            # Skip unrealistically large boxes (full-frame false positives)
            x, y, w, h = entry["box"]
            fh, fw = frameL.shape[:2]
            if w > 0.9 * fw and h > 0.9 * fh:
                continue
 
            if last_depth_map is not None:
                dist = get_object_distance(
                    last_depth_map, entry["box"], label=entry["label"])
                if dist is not None:
                    entry["distance"] = dist
                    objects_with_dist.append(entry)
                    continue
 
            objects_no_dist.append(entry)
 
        # Keep only 3 closest
        objects_with_dist.sort(key=lambda o: o["distance"])
        objects_with_dist = objects_with_dist[:3]
 
        print("\n--- PRIORITISED OBJECTS ---")
        for obj in objects_with_dist:
            print(f"{obj['label']} | {obj['distance']:.2f}m | conf={obj.get('confidence',0):.2f}")
 
        for obj in objects_with_dist:
            x, y, w, h = obj["box"]
            dist  = obj["distance"]
            conf = obj.get("confidence", 0)
 
            text  = f"{obj['label']} ({conf:.2f}) {dist:.2f}m"
 
            if dist < ALERT_DISTANCE:
                notify(obj["label"], dist)
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(overlay, text, (x, max(y-10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
 
        for obj in objects_no_dist:
            x, y, w, h = obj["box"]
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 200, 200), 2)
            conf = obj.get("confidence", 0)
            cv2.putText(overlay, f"{obj['label']} ({conf:.2f}) --m",
                        (x, max(y-10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 2)
 
        for obj in last_stairs:
            x, y, w, h = obj["box"]
            dist = None
            if last_depth_map is not None:
                dist = get_object_distance(
                    last_depth_map, obj["box"], label="stairs")
 
            if dist is not None:
                conf = obj.get("confidence", 0)
                text = f"STAIRS ({conf:.2f}) {dist:.2f}m"
                if dist < 2.0:
                    notify("stairs", dist)
            else:
                conf = obj.get("confidence", 0)
                text = f"STAIRS ({conf:.2f})"
 
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(overlay, text, (x, max(y-10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
 
        cv2.imshow("Stereo Vision System", overlay)
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
finally:
    camL.stop()
    camR.stop()
    cv2.destroyAllWindows()