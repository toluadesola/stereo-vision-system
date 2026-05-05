import cv2
import numpy as np
import glob
 
CHECKERBOARD = (9, 6)
SQUARE_SIZE  = 0.019 
 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0],
                        0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE
 
objpoints  = []
imgpointsL = []
imgpointsR = []
 
imagesL = sorted(glob.glob("calibration_images/left/*.jpg"))
imagesR = sorted(glob.glob("calibration_images/right/*.jpg"))
 
if len(imagesL) == 0:
    raise FileNotFoundError("No calibration images found in calibration_images/left/")
 
print(f"Found {len(imagesL)} left / {len(imagesR)} right images.")
 
image_size = None
 
for imgL_path, imgR_path in zip(imagesL, imagesR):
 
    imgL = cv2.imread(imgL_path)
    imgR = cv2.imread(imgR_path)
 
    # cv2.imread always returns BGR ? correct for COLOR_BGR2GRAY
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
 
    if image_size is None:
        image_size = (grayL.shape[1], grayL.shape[0])
 
    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)
 
    if retL and retR:
        objpoints.append(objp)
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)
        print(f"  Accepted: {imgL_path}")
    else:
        print(f"  Skipped (board not detected in both): {imgL_path}")
 
print(f"\n{len(objpoints)} pairs accepted for calibration.")
if len(objpoints) < 10:
    print("WARNING: fewer than 10 good pairs ? accuracy will be poor. Recapture more images.")
 
rmsL, mtxL, distL, _, _ = cv2.calibrateCamera(
    objpoints, imgpointsL, image_size, None, None)
rmsR, mtxR, distR, _, _ = cv2.calibrateCamera(
    objpoints, imgpointsR, image_size, None, None)
print(f"\nLeft  camera RMS reprojection error: {rmsL:.4f} px  (target < 0.5)")
print(f"Right camera RMS reprojection error: {rmsR:.4f} px  (target < 0.5)")
 

stereo_flags = (cv2.CALIB_USE_INTRINSIC_GUESS +
                cv2.CALIB_RATIONAL_MODEL)
 
rms_stereo, mtxL, distL, mtxR, distR, R, T, _, _ = cv2.stereoCalibrate(
    objpoints,
    imgpointsL,
    imgpointsR,
    mtxL, distL,
    mtxR, distR,
    image_size,
    flags=stereo_flags
)
print(f"Stereo RMS reprojection error:       {rms_stereo:.4f} px  (target < 1.0)")
baseline_m = np.linalg.norm(T)
print(f"Measured baseline:                   {baseline_m*100:.2f} cm  (expected ~8 cm)")
if abs(baseline_m - 0.08) > 0.02:
    print("  WARNING: baseline differs from expected 8 cm by more than 2 cm.")
    print("  This usually means the board moved between L and R captures.")
    print("  Delete calibration_images/ and recapture with a stationary board.")
 
# ?? Stereo rectification ??????????????????????????????????????????????????????
R1, R2, P1, P2, Q, roi_L, roi_R = cv2.stereoRectify(
    mtxL, distL,
    mtxR, distR,
    image_size,
    R, T,
    alpha=0          # alpha=0: crop to only valid (non-black) pixels
)
 
map1_L, map2_L = cv2.initUndistortRectifyMap(
    mtxL, distL, R1, P1, image_size, cv2.CV_16SC2)
map1_R, map2_R = cv2.initUndistortRectifyMap(
    mtxR, distR, R2, P2, image_size, cv2.CV_16SC2)
 

import os
os.makedirs("calibration", exist_ok=True)
np.savez("calibration/stereo_calibration.npz",
         map1_L=map1_L, map2_L=map2_L,
         map1_R=map1_R, map2_R=map2_R,
         Q=Q)
print("\nCalibration saved to calibration/stereo_calibration.npz")
print()
print("Next steps:")
print("  1. If stereo RMS < 1.0 px and baseline ~8 cm:")
print("     Open stereo.py and DELETE these two lines:")
print("       PHYSICAL_BASELINE_M = 0.08")
print("       self.Q[3, 2] = -1.0 / PHYSICAL_BASELINE_M")
print("     The Q matrix is now correct on its own.")
print()
print("  2. If stereo RMS is still > 1.0 px:")
print("     Run filter_calibration.py to remove bad pairs, then re-run.")
 
