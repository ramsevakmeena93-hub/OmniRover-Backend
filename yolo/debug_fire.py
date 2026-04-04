"""
Run this to see what HSV values your camera captures for fire.
Point your lighter at the camera, then press SPACE to capture and analyze.
Press Q to quit.
"""
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
print("Point lighter at camera, press SPACE to analyze frame, Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera - Press SPACE to analyze fire", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]

        # Sample center region
        cy, cx = h//2, w//2
        region = hsv[cy-50:cy+50, cx-50:cx+50]

        print("\n=== FRAME ANALYSIS ===")
        print(f"Frame size: {w}x{h}")
        print(f"Center HSV (H,S,V): {np.mean(region, axis=(0,1)).astype(int)}")

        # Find brightest orange-red pixels
        mask_loose = cv2.inRange(hsv, np.array([0,100,100]), np.array([30,255,255]))
        pixels = hsv[mask_loose > 0]
        if len(pixels) > 0:
            print(f"\nOrange-red pixels found: {len(pixels)}")
            print(f"H range: {pixels[:,0].min()} - {pixels[:,0].max()}")
            print(f"S range: {pixels[:,1].min()} - {pixels[:,1].max()}")
            print(f"V range: {pixels[:,2].min()} - {pixels[:,2].max()}")
            print(f"% of frame: {len(pixels)/(h*w)*100:.2f}%")
        else:
            print("No orange-red pixels found in frame")

        # BGR values of brightest spot
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, max_val, _, max_loc = cv2.minMaxLoc(gray)
        print(f"\nBrightest pixel at {max_loc}: BGR={frame[max_loc[1],max_loc[0]]}, HSV={hsv[max_loc[1],max_loc[0]]}")
        print("======================\n")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
