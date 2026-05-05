import cv2
import numpy as np
from config import *
 
PHYSICAL_BASELINE_M = 0.08
 
 
class StereoDepth:
 
    def __init__(self, calib_file):
 
        data = np.load(calib_file)
 
        self.map1_L = data["map1_L"]
        self.map2_L = data["map2_L"]
        self.map1_R = data["map1_R"]
        self.map2_R = data["map2_R"]
        self.Q      = data["Q"].copy()
 
        # Correct Q matrix baseline to physical measurement
        self.Q[3, 2] = -1.0 / PHYSICAL_BASELINE_M
 
        # SGBM left->right (primary matcher)
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=NUM_DISPARITIES,
            blockSize=BLOCK_SIZE,
            P1=8  * 3 * BLOCK_SIZE ** 2,
            P2=32 * 3 * BLOCK_SIZE ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
 
        # Right->left matcher needed for WLS filter
        # WLS (Weighted Least Squares) post-processes the disparity map to:
        #   - fill holes where SGBM found no match (common on plain surfaces)
        #   - fix wrong small disparities at longer distances (fixes doubling >1m)
        #   - smooth noise while preserving object edges
        self.stereo_r = cv2.ximgproc.createRightMatcher(self.stereo)
 
        # WLS filter parameters
        # lmbda: smoothness strength (higher = smoother edges)
        # sigma:  edge sensitivity (0.8-2.0 typical; lower preserves more detail)
        self.wls = cv2.ximgproc.createDisparityWLSFilter(self.stereo)
        self.wls.setLambda(8000)
        self.wls.setSigmaColor(1.2)
 
    def get_depth(self, frameL, frameR, return_disparity=False):
 
        rectL = cv2.remap(frameL, self.map1_L, self.map2_L, cv2.INTER_LINEAR)
        rectR = cv2.remap(frameR, self.map1_R, self.map2_R, cv2.INTER_LINEAR)
 
        grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
 
        # DISPARITY COMPUTATION
        disp_L = self.stereo.compute(grayL, grayR)
        disp_R = self.stereo_r.compute(grayR, grayL)
 
        # WLS filtering
        filtered_disp = self.wls.filter(disp_L, grayL, None, disp_R)
 
        # Convert to float disparity
        disparity = filtered_disp.astype(np.float32) / 16.0
 
        # Clean disparity
        valid_mask = disparity > 0
        disparity_clean = disparity.copy()
        disparity_clean[~valid_mask] = 0
 
        # DEPTH FROM DISPARITY
        points = cv2.reprojectImageTo3D(disparity_clean, self.Q)
        depth  = np.abs(points[:, :, 2])
 
        # Clean depth
        depth[~valid_mask] = np.nan
        depth[depth < 0.1]  = np.nan
        depth[depth > 10.0] = np.nan
 
        # RETURN BOTH (optional)
        if return_disparity:
            return depth, disparity_clean
 
        return depth