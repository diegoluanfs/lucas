import os

import cv2
from tkinter import filedialog, messagebox


def load_media_file(app):
    """Load an image or video file into the app and update the player state."""
    path = filedialog.askopenfilename(
        title="Selecionar Mídia",
        filetypes=[
            ("Vídeos", "*.avi *.mp4 *.mkv *.mov"),
            ("Imagens", "*.jpg *.png *.jpeg *.bmp"),
        ],
    )

    if not path:
        return

    app.video_path = path
    app.stop_video()
    app.img_result_original = None
    app.zoom_factor = 1.0
    app.roi_zoom = None

    if path.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
        app.cap = None
        frame = cv2.imread(path)
        if frame is not None:
            app.display_frame(frame)
            app.lbl_status.configure(text=f"Imagem: {os.path.basename(path)}")
    else:
        app.cap = cv2.VideoCapture(path)
        if app.cap is not None and app.cap.isOpened():
            total_frames = int(app.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if hasattr(app, "video_slider"):
                app.video_slider.configure(to=max(0, total_frames - 1))
            app.total_frames = total_frames
            app.current_frame_idx = 0
            app.show_frame()
            app.lbl_status.configure(text=f"Vídeo: {os.path.basename(path)}")
        else:
            messagebox.showerror("Erro", "Não foi possível abrir o arquivo de vídeo.")


def show_current_frame(app):
    """Render the current frame of the active video on the canvas."""
    if app.cap:
        app.cap.set(cv2.CAP_PROP_POS_FRAMES, app.current_frame_idx)
        ret, frame = app.cap.read()
        if ret:
            if app.analysis_active:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(frame, [max(contours, key=cv2.contourArea)], -1, (0, 255, 0), 2)
            app.display_frame(frame)


def toggle_video_player(app):
    """Toggle play/pause for the loaded video."""
    if not app.cap:
        return

    if hasattr(app, "video_slider_var"):
        app.cap.set(cv2.CAP_PROP_POS_FRAMES, int(app.video_slider_var.get()))

    app.video_playing = not app.video_playing
    app.btn_play.configure(text="PAUSE" if app.video_playing else "START")
    if app.video_playing:
        run_video_loop(app)


def run_video_loop(app):
    """Advance the loaded media through time while the play flag is enabled."""
    if app.video_playing and app.cap:
        if app.current_frame_idx < app.total_frames - 1:
            app.current_frame_idx += 1
            app.show_frame()
            app.after_id = app.after(30, lambda: run_video_loop(app))
        else:
            app.video_playing = False
            app.btn_play.configure(text="START")


def stop_video_player(app):
    """Stop video playback and clear any scheduling loop."""
    app.video_playing = False
    if app.after_id:
        app.after_cancel(app.after_id)
        app.after_id = None
    if hasattr(app, "btn_play"):
        app.btn_play.configure(text="START")


def prev_frame(app):
    """Go to the previous frame of the current video."""
    if app.current_frame_idx > 0:
        app.current_frame_idx -= 1
        app.show_frame()


def next_frame(app):
    """Go to the next frame of the current video."""
    if app.current_frame_idx < app.total_frames - 1:
        app.current_frame_idx += 1
        app.show_frame()
