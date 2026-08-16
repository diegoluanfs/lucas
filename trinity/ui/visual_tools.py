import cv2
import numpy as np
from tkinter import filedialog, messagebox

from trinity.analysis.static_angle import run_circle_fit


def edge_detection(app):
    """Apply Canny and contour detection to highlight the droplet silhouette."""
    if app.img_result_original is None:
        messagebox.showwarning("Warning", "Upload an image first!")
        return

    app.show_edges = not app.show_edges
    gray = cv2.cvtColor(app.img_result_original, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    v = np.median(blur)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edged = cv2.Canny(blur, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)

    result_vis = app.img_result_original.copy()
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours:
        c_max = max(contours, key=cv2.contourArea)
        cv2.drawContours(result_vis, [c_max], -1, (0, 255, 0), 2)
        app.current_contour = c_max
        app.display_frame(result_vis, update_original=False)
        app.lbl_status.configure(text="Edges successfully detected.")
    else:
        app.lbl_status.configure(text="No outline found.")
        app.display_frame(app.img_result_original, update_original=False)
        app.lbl_status.configure(text="Border Preview: DISABLED")


def show_important_info(app):
    """Display key guidance directly on the canvas."""
    app.display_frame(app.img_result_original)


def save_current_image(app):
    """Save the image currently displayed in the app canvas."""
    if app.img_result_original is None:
        messagebox.showwarning("Warning", "There is no image to save!")
        return

    img_to_save = getattr(app, 'current_vis_img', app.img_result_original)

    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
        title="Save Measurement Image",
    )

    if file_path:
        try:
            cv2.imwrite(file_path, img_to_save)
            messagebox.showinfo("Success", f"Image saved in:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {e}")


def fitting(app):
    """Run the circular-fit routine through the shared static-angle helper."""
    return run_circle_fit(app)
