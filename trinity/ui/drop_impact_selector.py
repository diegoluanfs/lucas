import cv2


def select_manual_floor_line(video_path):
    """Select a video frame and two points defining the inclined baseline."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = 0
    points = []
    line_params = {"p1": (0, 300), "p2": (1000, 300)}
    window_name = "A/D: Frame | 2 Clicks: Define Inclined Line | ENTER: Confirm"

    def get_frame(index):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        return frame if ret else None

    frame = get_frame(frame_index)
    if frame is None:
        cap.release()
        return line_params
    display_frame = frame.copy()

    def click_event(event, x, y, _flags, _param):
        nonlocal display_frame, points
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        points.append((x, y))
        cv2.circle(display_frame, (x, y), 4, (0, 255, 0), -1)
        if len(points) == 2:
            display_frame = frame.copy()
            cv2.line(display_frame, points[0], points[1], (0, 0, 255), 2)
            line_params["p1"], line_params["p2"] = points
            points = []
        cv2.imshow(window_name, display_frame)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event)

    try:
        while True:
            visible = display_frame.copy()
            cv2.putText(
                visible,
                f"Frame: {frame_index}/{total_frames - 1}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow(window_name, visible)
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("d"), ord("D"), ord("a"), ord("A")):
                if key in (ord("d"), ord("D")) and frame_index < total_frames - 1:
                    frame_index += 1
                elif key in (ord("a"), ord("A")) and frame_index > 0:
                    frame_index -= 1
                frame = get_frame(frame_index)
                if frame is not None:
                    display_frame = frame.copy()
                    cv2.line(display_frame, line_params["p1"], line_params["p2"], (0, 0, 255), 2)
            elif key == 13 or key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyWindow(window_name)

    return line_params
