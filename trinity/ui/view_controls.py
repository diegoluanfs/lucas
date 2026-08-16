import cv2
import numpy as np
from PIL import Image
import customtkinter as ctk

from trinity.analysis.drop_impact import robust_segmentation as segment_drop_background


def apply_zoom(app, factor):
    app.zoom_factor *= factor
    if app.zoom_factor < 1.0:
        app.zoom_factor = 1.0
        app.roi_zoom = None

    if app.img_result_original is not None:
        app.display_frame(app.img_result_original, update_original=False)


def on_mouse_wheel(app, event):
    if app.img_result_original is not None:
        if event.delta > 0 or event.num == 4:
            apply_zoom(app, 1.1)
        else:
            apply_zoom(app, 0.9)


def on_mouse_scroll_windows(app, event):
    """Handle mouse wheel zoom on Windows."""
    if event.delta > 0:
        apply_zoom(app, 1.1)
    elif event.delta < 0:
        apply_zoom(app, 0.9)


def on_click_press(app, event):
    if app.img_result_original is not None:
        app.dragging = True
        app.start_x, app.start_y = event.x, event.y


def on_mouse_drag(app, event):
    pass


def on_click_release(app, event):
    if app.dragging and app.img_result_original is not None:
        app.dragging = False
        end_x, end_y = event.x, event.y
        if abs(end_x - app.start_x) > 10 and abs(end_y - app.start_y) > 10:
            app.roi_zoom = (app.start_x, app.start_y, end_x, end_y)
            update_zoom_view(app, is_roi=True)


def update_zoom_view(app, is_roi=False):
    if app.img_result_original is not None:
        app.display_frame(app.img_result_original, update_original=False)


def display_frame(app, frame, update_original=True):
    if frame is None:
        return

    if app.cap and hasattr(app, 'video_slider_var'):
        current_idx = int(app.cap.get(cv2.CAP_PROP_POS_FRAMES))
        app._syncing_slider = True
        try:
            app.video_slider_var.set(current_idx)
        finally:
            app._syncing_slider = False

    if update_original:
        app.img_result_original = frame.copy()

    if not hasattr(app, 'zoom_factor'):
        app.zoom_factor = 1.0
    if not hasattr(app, 'roi_zoom'):
        app.roi_zoom = None

    h, w = frame.shape[:2]

    if app.roi_zoom is not None:
        tw = app.canvas_view.winfo_width()
        th = app.canvas_view.winfo_height()
        if tw > 0 and th > 0:
            x1, y1, x2, y2 = app.roi_zoom
            rx1 = int(min(x1, x2) * (w / tw))
            ry1 = int(min(y1, y2) * (h / th))
            rx2 = int(max(x1, x2) * (w / tw))
            ry2 = int(max(y1, y2) * (h / th))

            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(w, rx2), min(h, ry2)

            if rx2 > rx1 and ry2 > ry1:
                frame = frame[ry1:ry2, rx1:rx2]

    elif app.zoom_factor > 1.0:
        new_w, new_h = int(w / app.zoom_factor), int(h / app.zoom_factor)
        x1, y1 = max(0, (w - new_w) // 2), max(0, (h - new_h) // 2)
        frame = frame[y1:y1 + new_h, x1:x1 + new_w]

    app.update_idletasks()

    h, w = frame.shape[:2]
    app.current_vis_img = frame.copy()

    window_w = app.winfo_width()
    window_h = app.winfo_height()

    max_allowed_w = window_w - 300
    max_allowed_h = window_h - 250

    if max_allowed_w <= 100:
        max_allowed_w = 800
    if max_allowed_h <= 100:
        max_allowed_h = 500

    ratio = min(max_allowed_w / w, max_allowed_h / h)
    new_size = (int(w * ratio), int(h * ratio))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=new_size)

    app.canvas_view.configure(image=img_tk, text="")
    app.canvas_view.image = img_tk


def on_slider_move(app, value):
    """Update the video display to the frame selected by the slider."""
    if app.cap and not getattr(app, '_syncing_slider', False):
        frame_destino = int(float(value))
        app.current_frame_idx = frame_destino
        app.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_destino)

        ret, frame = app.cap.read()
        if ret:
            app.img_result_original = frame.copy()
            app.display_frame(frame, update_original=False)
            app.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_destino)


def robust_segmentation(app, frame_gray, background):
    """Background subtraction and droplet enhancement for motion analysis."""
    return segment_drop_background(frame_gray, background)
