import cv2
import numpy as np


def detect_edges(frame):
    """Detect and draw the largest external contour on a BGR frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    median = np.median(blur)
    lower = int(max(0, (1.0 - 0.33) * median))
    upper = int(min(255, (1.0 + 0.33) * median))
    edged = cv2.Canny(blur, lower, upper)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

    result = frame.copy()
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return result, None

    contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
    return result, contour
