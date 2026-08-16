import cv2


def select_three_points(image):
    """Select two baseline points and one angle point in an image ROI."""
    points = []
    preview = image.copy()
    window_name = "Select 3 Points (Base 1, Base 2, Angle)"
    mouse_position = [0, 0]

    def handle_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE:
            mouse_position[0], mouse_position[1] = x, y

    cv2.imshow(window_name, preview)
    cv2.setMouseCallback(window_name, handle_mouse)

    try:
        while len(points) < 3:
            display = preview.copy()
            for point in points:
                cv2.circle(display, point, 1, (0, 0, 255), -1)

            if len(points) == 1:
                cv2.line(display, points[0], tuple(mouse_position), (0, 255, 255), 1)
            elif len(points) == 2:
                cv2.line(display, points[0], points[1], (0, 0, 255), 2)
                cv2.line(display, points[1], tuple(mouse_position), (0, 255, 255), 1)

            cv2.imshow(window_name, display)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cv2.destroyWindow(window_name)

    return points if len(points) == 3 else None
