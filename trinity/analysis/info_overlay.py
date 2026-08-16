import cv2
import numpy as np


HELP_TEXT = [
    "IMPORTANT INFORMATION:",
    "- All videos should start without the drop appearing.",
    "- Select the method that best suits your videos and images.",
    "- In some analyses, select the area that includes the droplet (ROI).",
    "- The baseline must be selected horizontally in the interface.",
    "- Adjust the parameters if you notice any errors in the analysis.",
    "",
    "Click Home or Import to return.",
]


def create_info_overlay(image=None):
    """Create the instructional image shown by the information action."""
    if image is None:
        info_image = np.zeros((600, 800, 3), dtype=np.uint8)
    else:
        info_image = cv2.addWeighted(image.copy(), 0.2, image, 0, 0)

    for index, text in enumerate(HELP_TEXT):
        y = 50 + index * 40
        cv2.putText(
            info_image,
            text,
            (50, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return info_image
