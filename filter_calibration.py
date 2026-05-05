"""
Analyses every calibration image pair and ranks them by reprojection error.
Automatically moves bad pairs (above the threshold) to a rejected/ folder
so you can re-run calibration.py with only the good ones.
 
Run this BEFORE calibration.py when you have a large set of images.
 
Usage:
    python filter_calibration.py
    python calibration.py          # re-run with cleaned image set
"""
 
import cv2
import numpy as np
import glob
import os
import shutil
 
CHECKERBOARD = (9, 6)
SQUARE_SIZE  = 0.019   # must match calibration.py
 
# Pairs with per-image RMS above this are moved to rejected/
# Start at 0.5 and lower if RMS is still high after filtering
ERROR_THRESHOLD = 0.1
 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0],
                        0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE
 
# Collect all valid pairs
imagesL = sorted(glob.glob("calibration_images/left/*.jpg"))
imagesR = sorted(glob.glob("calibration_images/right/*.jpg"))
 
objpoints  = []
imgpointsL = []
imgpointsR = []
valid_pairs = []   # (left_path, right_path) for accepted pairs
image_size  = None
 
print(f"Found {len(imagesL)} left / {len(imagesR)} right images.\n")
 
for imgL_path, imgR_path in zip(imagesL, imagesR):
    imgL  = cv2.imread(imgL_path)
    imgR  = cv2.imread(imgR_path)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
 
    if image_size is None:
        image_size = (grayL.shape[1], grayL.shape[0])
 
    retL, cL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
    retR, cR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)
 
    if retL and retR:
        cL = cv2.cornerSubPix(grayL, cL, (11,11), (-1,-1), criteria)
        cR = cv2.cornerSubPix(grayR, cR, (11,11), (-1,-1), criteria)
        objpoints.append(objp)
        imgpointsL.append(cL)
        imgpointsR.append(cR)
        valid_pairs.append((imgL_path, imgR_path))
 
print(f"{len(valid_pairs)} pairs detected.\n")
 
# Calibrate each camera to get rvecs/tvecs
_, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(
    objpoints, imgpointsL, image_size, None, None)
_, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(
    objpoints, imgpointsR, image_size, None, None)
 
# Compute per-image reprojection error
errors = []
for i in range(len(objpoints)):
    proj_L, _ = cv2.projectPoints(objpoints[i], rvecsL[i], tvecsL[i], mtxL, distL)
    proj_R, _ = cv2.projectPoints(objpoints[i], rvecsR[i], tvecsR[i], mtxR, distR)
    errL = cv2.norm(imgpointsL[i], proj_L, cv2.NORM_L2) / len(proj_L)
    errR = cv2.norm(imgpointsR[i], proj_R, cv2.NORM_L2) / len(proj_R)
    err  = (errL + errR) / 2
    errors.append((err, i))
 
errors.sort(key=lambda x: x[0])
 
# Report
print(f"{'Rank':<5} {'Error (px)':<12} {'Status':<10} Left image")
print("-" * 65)
keep_count   = 0
reject_count = 0
to_reject    = []
 
for rank, (err, idx) in enumerate(errors):
    lpath, rpath = valid_pairs[idx]
    status = "KEEP" if err <= ERROR_THRESHOLD else "REJECT"
    if status == "KEEP":
        keep_count += 1
    else:
        reject_count += 1
        to_reject.append((lpath, rpath))
    print(f"{rank+1:<5} {err:<12.4f} {status:<10} {os.path.basename(lpath)}")
 
print(f"\nKeeping {keep_count} pairs, rejecting {reject_count} pairs "
      f"(threshold = {ERROR_THRESHOLD} px)")
 
# Move bad pairs to rejected
if to_reject:
    os.makedirs("calibration_images/rejected/left",  exist_ok=True)
    os.makedirs("calibration_images/rejected/right", exist_ok=True)
 
    for lpath, rpath in to_reject:
        fname = os.path.basename(lpath)
        shutil.move(lpath, f"calibration_images/rejected/left/{fname}")
        shutil.move(rpath, f"calibration_images/rejected/right/{fname}")
 
    print(f"\nMoved {len(to_reject)} bad pairs to calibration_images/rejected/")
    print("Now run:  python calibration.py")
else:
    print("\nAll pairs are within threshold ? no images removed.")
    print("If RMS is still high in calibration.py, lower ERROR_THRESHOLD to 0.3")
 