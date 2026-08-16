import cv2
import numpy as np
from scipy.signal import savgol_filter


def robust_segmentation(frame_gray, background):
    """Segment a moving droplet by subtracting a reference background."""
    diff = cv2.absdiff(background, frame_gray)
    blurred = cv2.medianBlur(diff, 5)
    _, mask = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)


def segment_velocity_drop(diff_frame, line_a, line_b):
    """Build the post-impact edge mask for the first velocity method."""
    blur = cv2.GaussianBlur(diff_frame, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    height, width = edges.shape
    for x in range(width):
        y_limit = int(line_a * x + line_b)
        if y_limit < height:
            edges[y_limit:, x] = 0
    return edges


def segment_velocity_drop_v2(diff_frame, line_b):
    """Build the post-impact edge mask for the second velocity method."""
    blur = cv2.GaussianBlur(diff_frame, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges[int(line_b):, :] = 0
    return edges


def process_velocity_frame(app):
    """Process one frame of the first Drop Impact velocity method."""
    if not app.vel_active or app.cap_vel is None:
        return

    ret, frame = app.cap_vel.read()
    if not ret:
        params, data = app.vel_params, app.vel_data
        if len(data["posicoes"]) > 2:
            times = np.array(data["tempos"])
            positions_m = np.array(data["posicoes"]) / 1000.0
            impact_velocity, _ = np.polyfit(times, positions_m, 1)
            mean_diameter_mm = np.mean(data["pre_impacto"]) if data["pre_impacto"] else 1.0
            diameter_m = mean_diameter_mm / 1000.0
            reynolds = (params["densidade"] * abs(impact_velocity) * diameter_m) / params["viscosidade"]
            weber = (params["densidade"] * (impact_velocity**2) * diameter_m) / params["tensao"]
            ohnesorge = params["viscosidade"] / np.sqrt(params["densidade"] * params["tensao"] * diameter_m)
            mean_area = sum(data["areas_queda"]) / len(data["areas_queda"]) if data.get("areas_queda") else 0.0
            app.mostrar_resultados_adimensionais(abs(impact_velocity), mean_diameter_mm, reynolds, weber, ohnesorge, mean_area)

        app.cap_vel.release()
        app.vel_active = False
        app.lbl_status.configure(text="Drop Impact analysis completed.")
        return

    params, state, data = app.vel_params, app.vel_state, app.vel_data
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_masked = gray.copy()
    gray_masked[gray_masked < params["gray_thresh"]] = 0

    if state["background"] is None:
        state["background"] = gray_masked.copy()
        state["frame_id"] += 1
        app.after(1, app.processar_frame_velocidade)
        return

    diff = cv2.absdiff(gray_masked, state["background"])
    middle_y = int(params["line_a"] * (frame.shape[1] / 2) + params["line_b"])

    if not state["impacto"]:
        roi_limit = max(50, middle_y - 50)
        blur_roi = cv2.GaussianBlur(diff[0:roi_limit, :], (5, 5), 0)
        _, falling_mask = cv2.threshold(blur_roi, app.threshold_impact, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(falling_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            area_px = cv2.contourArea(contour)
            if area_px > params["area_min_queda"]:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                diameter_mm = (2 * radius) * params["escala"]
                data["tempos"].append(state["frame_id"] * params["dt"])
                data["posicoes"].append(y * params["escala"])
                data["pre_impacto"].append(diameter_mm)
                data["areas_queda"].append(area_px)
                if len(data["pre_impacto"]) > params["frames_antes"]:
                    state["diametro_pre"] = data["pre_impacto"][-int(params["frames_antes"])]
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                if y + radius >= params["line_a"] * x + params["line_b"]:
                    state["impacto"] = True
    else:
        mask = segment_velocity_drop(diff, params["line_a"], params["line_b"])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        candidates = [contour for contour in contours if cv2.contourArea(contour) > params["area_min_pos"]]
        if candidates:
            droplet = max(candidates, key=cv2.contourArea)
            points = droplet.reshape(-1, 2)
            smooth_points = points if len(points) <= 21 else np.column_stack((savgol_filter(points[:, 0], 21, 3), savgol_filter(points[:, 1], 21, 3))).astype(np.int32)
            baseline_y = params["line_a"] * smooth_points[:, 0] + params["line_b"]
            contacts = smooth_points[np.abs(smooth_points[:, 1] - baseline_y) <= params["tolerancia"]]
            if len(contacts) > 2:
                left_index, right_index = np.argmin(contacts[:, 0]), np.argmax(contacts[:, 0])
                point_left, point_right = contacts[left_index], contacts[right_index]
                distance_px = np.sqrt((point_right[0] - point_left[0])**2 + (point_right[1] - point_left[1])**2)
                diameter_mm = distance_px * params["escala"]
                beta = diameter_mm / state["diametro_pre"]
                elapsed = state["frame_id"] * params["dt"]
                area_mm2 = cv2.contourArea(droplet) * (params["escala"]**2)
                center_x = (point_left[0] + point_right[0]) / 2
                center_y = (point_left[1] + point_right[1]) / 2
                base_center = (int(center_x), int(center_y))
                top_point = smooth_points[np.argmin(np.abs(smooth_points[:, 0] - center_x))]
                height_mm = np.sqrt((top_point[0] - center_x)**2 + (top_point[1] - center_y)**2) * params["escala"]
                previous_diameter = state.get("last_diam_val")
                diameter_rate = (diameter_mm - previous_diameter) / params["dt"] if previous_diameter is not None else 0.0
                state["last_diam_val"] = diameter_mm
                cv2.line(frame, tuple(top_point), base_center, (255, 0, 255), 2)
                cv2.circle(frame, tuple(top_point), 3, (255, 0, 255), -1)
                app.tree.insert("", "end", values=(f"{elapsed:.4f}", "-", f"{diameter_mm:.2f}", f"{beta:.2f}", f"{height_mm:.2f}", f"{area_mm2:.2f}", f"{diameter_rate:.2f}"))
                cv2.polylines(frame, [smooth_points.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
                cv2.line(frame, tuple(point_left), tuple(point_right), (0, 0, 255), 2)

    height, width = frame.shape[:2]
    cv2.line(frame, (0, int(params["line_b"])), (width, int(params["line_a"] * width + params["line_b"])), (255, 0, 0), 1)
    state["frame_id"] += 1
    app.display_frame(frame)
    app.after(1, app.processar_frame_velocidade)


def process_velocity_frame_v2(app):
    """Process one frame of the second Drop Impact velocity method."""
    if not app.vel_active or app.cap_vel is None:
        return

    ret, frame = app.cap_vel.read()
    if not ret:
        params, data = app.vel_params, app.vel_data
        if len(data["posicoes"]) > 2:
            times = np.array(data["tempos"])
            positions_m = np.array(data["posicoes"]) / 1000.0
            impact_velocity, _ = np.polyfit(times, positions_m, 1)
            mean_diameter_mm = np.mean(data["pre_impacto"]) if data["pre_impacto"] else 1.0
            diameter_m = mean_diameter_mm / 1000.0
            reynolds = (params["densidade"] * abs(impact_velocity) * diameter_m) / params["viscosidade"]
            weber = (params["densidade"] * (abs(impact_velocity)**2) * diameter_m) / params["tensao"]
            ohnesorge = params["viscosidade"] / np.sqrt(params["densidade"] * params["tensao"] * diameter_m)
            mean_area = sum(data["areas_queda"]) / len(data["areas_queda"]) if data["areas_queda"] else 0.0
            app.mostrar_resultados_adimensionais(abs(impact_velocity), mean_diameter_mm, reynolds, weber, ohnesorge, mean_area)
        app.cap_vel.release()
        app.vel_active = False
        app.lbl_status.configure(text="Análise de impacto concluída.")
        return

    params, state, data = app.vel_params, app.vel_state, app.vel_data
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if state["background"] is None:
        state["background"] = frame_gray.copy()
        state["frame_id"] += 1
        app.after(1, app.processar_frame_velocidade_2)
        return

    diff = cv2.absdiff(frame_gray, state["background"])
    if not state["impacto"]:
        roi_height = int(params["line_b"]) - 10 if "line_b" in params else 250
        blur = cv2.GaussianBlur(diff[:roi_height, :], (5, 5), 0)
        _, falling_mask = cv2.threshold(blur, app.threshold_impact, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(falling_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            area_px = cv2.contourArea(contour)
            if area_px > params["area_min_queda"]:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                diameter_mm = (2 * radius) * params["escala"]
                data["areas_queda"].append(area_px)
                data["tempos"].append(state["frame_id"] * params["dt"])
                data["posicoes"].append(y * params["escala"])
                data["pre_impacto"].append(diameter_mm)
                if len(data["pre_impacto"]) > params["frames_antes"]:
                    state["diametro_pre"] = data["pre_impacto"][-int(params["frames_antes"])]
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                if y + radius >= params["line_a"] * x + params["line_b"]:
                    state["impacto"] = True
    else:
        mask = segment_velocity_drop_v2(diff, params["line_b"])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        candidates = [contour for contour in contours if cv2.contourArea(contour) > params["area_min_pos"]]
        if candidates:
            droplet = cv2.convexHull(max(candidates, key=cv2.contourArea))
            x, y, width, height = cv2.boundingRect(droplet)
            diameter_mm = width * params["escala"]
            height_mm = height * params["escala"]
            beta = diameter_mm / state["diametro_pre"]
            area_mm2 = cv2.contourArea(droplet) * (params["escala"]**2)
            previous_diameter = state.get("last_diam_val")
            diameter_rate = (diameter_mm - previous_diameter) / params["dt"] if previous_diameter is not None else 0.0
            state["last_diam_val"] = diameter_mm
            elapsed = state["frame_id"] * params["dt"]
            app.tree.insert("", "end", values=(f"{elapsed:.4f}", "-", f"{diameter_mm:.2f}", f"{beta:.2f}", f"{height_mm:.2f}", f"{area_mm2:.2f}", f"{diameter_rate:.2f}"))
            cv2.drawContours(frame, [droplet], -1, (0, 255, 0), 2)
            middle_y = int(y + height / 2)
            cv2.line(frame, (x, middle_y), (x + width, middle_y), (0, 0, 255), 2)
            cv2.line(frame, (int(x + width / 2), y), (int(x + width / 2), y + height), (255, 0, 0), 2)
            cv2.putText(frame, f"Beta: {beta:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    state["frame_id"] += 1
    app.display_frame(frame)
    app.after(1, app.processar_frame_velocidade_2)


def process_third_impact_frame(app):
    """Process one frame of the third Drop Impact method."""
    if not app.vel_active or app.cap_vel is None:
        return

    params, data = app.params, app.data
    ret, frame = app.cap_vel.read()
    if not ret:
        if len(data["posicoes"]) > 2:
            times = np.array(data["tempos"])
            positions_m = np.array(data["posicoes"]) / 1000.0
            impact_velocity, _ = np.polyfit(times, positions_m, 1)
            velocity = abs(impact_velocity)
            mean_diameter_mm = data["diam_pre"] if data["diam_pre"] > 0 else 1.0
            diameter_m = mean_diameter_mm / 1000.0
            reynolds = (params["densidade"] * velocity * diameter_m) / params["viscosidade"]
            weber = (params["densidade"] * (velocity**2) * diameter_m) / params["tensao"]
            ohnesorge = params["viscosidade"] / np.sqrt(params["densidade"] * params["tensao"] * diameter_m)
            mean_area = sum(data["areas_queda"]) / len(data["areas_queda"]) if data["areas_queda"] else 0
            app.mostrar_resultados_adimensionais(velocity, mean_diameter_mm, reynolds, weber, ohnesorge, mean_area)
        app.cap_vel.release()
        app.vel_active = False
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = robust_segmentation(gray, app.background_gray)
    mask[int(app.limite_y_base):, :] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        width = frame.shape[1]
        contour = min(contours, key=lambda item: abs(cv2.moments(item)["m10"] / (cv2.moments(item)["m00"] + 1e-5) - width / 2))
        area_px = cv2.contourArea(contour)
        if area_px > params["min_area"]:
            _, top_y, _, box_height = cv2.boundingRect(contour)
            bottom_y = top_y + box_height
            cv2.line(frame, (0, int(app.limite_y_base)), (width, int(app.limite_y_base)), (0, 255, 255), 1)
            if not app.impacto_detectado:
                data["areas_queda"].append(area_px)
                data["tempos"].append(app.frame_id * params["dt"])
                data["posicoes"].append(top_y * params["escala"])
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    data["diam_pre"] = ((ellipse[1][0] + ellipse[1][1]) / 2) * params["escala"]
                    cv2.ellipse(frame, ellipse, (255, 0, 0), 2)
                if abs(bottom_y - app.limite_y_base) < 5:
                    app.impacto_detectado = True
            else:
                hull = cv2.convexHull(contour)
                point_left = tuple(hull[hull[:, :, 0].argmin()][0])
                point_right = tuple(hull[hull[:, :, 0].argmax()][0])
                diameter_mm = (point_right[0] - point_left[0]) * params["escala"]
                beta = diameter_mm / data["diam_pre"] if data["diam_pre"] > 0 else 0
                area_mm2 = area_px * (params["escala"]**2)
                elapsed = app.frame_id * params["dt"]
                point_top = tuple(hull[hull[:, :, 1].argmin()][0])
                point_base = tuple(hull[hull[:, :, 1].argmax()][0])
                height_mm = (point_base[1] - point_top[1]) * params["escala"]
                previous_diameter = data.get("last_diam_val")
                diameter_rate = (diameter_mm - previous_diameter) / params["dt"] if previous_diameter is not None else 0.0
                data["last_diam_val"] = diameter_mm
                cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)
                cv2.line(frame, point_left, point_right, (0, 0, 255), 2)
                cv2.line(frame, (point_top[0], point_top[1]), (point_top[0], point_base[1]), (255, 0, 255), 2)
                app.tree.insert("", "end", values=(f"{elapsed:.5f}", "-", f"{diameter_mm:.3f}", f"{beta:.3f}", f"{height_mm:.3f}", f"{area_mm2:.3f}", f"{diameter_rate:.3f}"))

    app.display_frame(frame)
    app.frame_id += 1
    app.after(1, app.processar_frame_impacto_tres)


def process_fourth_impact_frame(app):
    """Process one frame of the fourth Drop Impact method."""
    if not app.vel_active or app.cap_vel is None or not getattr(app, "analysis_active", False):
        return

    params, data = app.params, app.data
    ret, frame = app.cap_vel.read()
    if not ret:
        if len(data["posicoes"]) > 2:
            times = np.array(data["tempos"])
            positions_m = np.array(data["posicoes"]) / 1000.0
            impact_velocity, _ = np.polyfit(times, positions_m, 1)
            velocity = abs(impact_velocity)
            mean_diameter_mm = data["diam_pre"] if data["diam_pre"] > 0 else 1.0
            diameter_m = mean_diameter_mm / 1000.0
            reynolds = (params["densidade"] * velocity * diameter_m) / params["viscosidade"]
            weber = (params["densidade"] * (velocity**2) * diameter_m) / params["tensao"]
            ohnesorge = params["viscosidade"] / np.sqrt(params["densidade"] * params["tensao"] * diameter_m)
            mean_area = sum(data["areas_queda"]) / len(data["areas_queda"]) if data["areas_queda"] else 0
            app.mostrar_resultados_adimensionais(velocity, mean_diameter_mm, reynolds, weber, ohnesorge, mean_area)
        app.cap_vel.release()
        app.vel_active = False
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if app.background_gray is not None and gray.shape == app.background_gray.shape:
        difference = cv2.absdiff(gray, app.background_gray)
        cleaned = cv2.bilateralFilter(difference, 5, 75, 75)
        _, mask = cv2.threshold(cleaned, 15, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        cleaned = cv2.bilateralFilter(frame, 5, 75, 75)
        gray_clean = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_clean, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    height, width = frame.shape[:2]
    for x in range(width):
        limit_y = int(app.m_base * x + app.c_base)
        if 0 <= limit_y < height:
            mask[limit_y:, x] = 0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_start = (0, int(app.c_base))
    line_end = (width, int(app.m_base * width + app.c_base))
    cv2.line(frame, line_start, line_end, (0, 255, 255), 1)

    if contours:
        contour = min(contours, key=lambda item: abs(cv2.moments(item)["m10"] / (cv2.moments(item)["m00"] + 1e-5) - width / 2))
        area_px = cv2.contourArea(contour)
        if area_px > params["min_area"]:
            _, top_y, _, box_height = cv2.boundingRect(contour)
            bottom_y = top_y + box_height
            hull = cv2.convexHull(contour)
            center_x = box_height + (box_height + box_height) // 2
            limit_at_center = app.m_base * center_x + app.c_base

            if not app.impacto_detectado:
                data["areas_queda"].append(area_px)
                data["tempos"].append(app.frame_id * params["dt"])
                data["posicoes"].append(top_y * params["escala"])
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    data["diam_pre"] = ((ellipse[1][0] + ellipse[1][1]) / 2) * params["escala"]
                    cv2.ellipse(frame, ellipse, (255, 0, 0), 2)
                if abs(bottom_y - limit_at_center) < 5:
                    app.impacto_detectado = True
            else:
                contact_points = []
                for point in contour:
                    px, py = point[0][0], point[0][1]
                    expected_y = app.m_base * px + app.c_base
                    if abs(py - expected_y) <= 3:
                        contact_points.append((px, int(expected_y)))

                if len(contact_points) >= 2:
                    contact_points.sort(key=lambda item: item[0])
                    point_left, point_right = contact_points[0], contact_points[-1]
                    diameter_px = np.sqrt((point_right[0] - point_left[0])**2 + (point_right[1] - point_left[1])**2)
                    diameter_mm = diameter_px * params["escala"]
                    beta = diameter_mm / data["diam_pre"] if data["diam_pre"] > 0 else 0
                    area_mm2 = area_px * (params["escala"]**2)
                    elapsed = app.frame_id * params["dt"]
                    top_point = tuple(hull[hull[:, :, 1].argmin()][0])
                    base_y_at_top = app.m_base * top_point[0] + app.c_base
                    height_mm = abs(base_y_at_top - top_point[1]) * params["escala"]
                    previous_diameter = data.get("last_diam_val")
                    diameter_rate = (diameter_mm - previous_diameter) / params["dt"] if previous_diameter is not None else 0.0
                    data["last_diam_val"] = diameter_mm
                    if diameter_px > data["max_w_px"]:
                        data["max_w_px"] = diameter_px
                    cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)
                    cv2.line(frame, point_left, point_right, (0, 0, 255), 2)
                    cv2.line(frame, top_point, (top_point[0], int(base_y_at_top)), (255, 0, 255), 2)
                    app.tree.insert("", "end", values=(f"{elapsed:.5f}", "-", f"{diameter_mm:.3f}", f"{beta:.3f}", f"{height_mm:.3f}", f"{area_mm2:.3f}", f"{diameter_rate:.3f}"))
                else:
                    elapsed = app.frame_id * params["dt"]
                    data["last_diam_val"] = None
                    cv2.drawContours(frame, [hull], -1, (0, 0, 255), 1)
                    app.tree.insert("", "end", values=(f"{elapsed:.5f}", "-", "0.000", "0.000", "-", "-", "-"))

    app.display_frame(frame, update_original=True)
    app.frame_id += 1
    app.after(1, app.processar_frame_impacto_quatro)
