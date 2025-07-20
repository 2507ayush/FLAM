import cv2
import numpy as np
import math
import os

def visualize_light_direction(bg_img, light_dir, output_path="light_direction_visualization.jpg"):
    vis_img = bg_img.copy()
    h, w = vis_img.shape[:2]
    center_x, center_y = w // 2, h // 2
    arrow_length = min(h, w) // 3
    end_x = int(center_x + light_dir[0] * arrow_length)
    end_y = int(center_y + light_dir[1] * arrow_length)
    cv2.arrowedLine(vis_img, (center_x, center_y), (end_x, end_y), (0, 0, 255), 5, tipLength=0.3)
    cv2.putText(vis_img, f"Light Direction: ({light_dir[0]:.2f}, {light_dir[1]:.2f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(output_path, vis_img)
    print(f"[INFO] Light direction visualization saved to: {output_path}")
    return vis_img

def detect_light_direction(bg_img, visualize=False):
    gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=40, maxLineGap=20)
    if lines is not None and len(lines) > 10:
        angles = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                angles.append(angle)
        avg_angle = np.mean(angles)
        dx = math.cos(math.radians(avg_angle))
        dy = math.sin(math.radians(avg_angle))
        print(f"[INFO] Light direction detected: dx={dx:.2f}, dy={dy:.2f}")
        if visualize:
            visualize_light_direction(bg_img, (dx, dy))
        return (dx, dy)
    print("[WARN] Not enough lines detected. Using default light direction.")
    default_dir = (1, 0.2)
    if visualize:
        visualize_light_direction(bg_img, default_dir)
    return default_dir

def sharpen_shadow(shadow_mask, intensity=0.5, radius=3):
    blurred = cv2.GaussianBlur(shadow_mask, (radius*2+1, radius*2+1), 0)
    sharpened = cv2.addWeighted(shadow_mask, 1.0 + intensity, blurred, -intensity, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    sharpened = np.where(shadow_mask > 0, sharpened, 0)
    return sharpened

def create_realistic_shadow(mask, light_dir, intensity=0.70):
    h, w = mask.shape
    dx = int(light_dir[0] * 20)
    dy = int(light_dir[1] * 20)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shadow = cv2.warpAffine(mask, M, (w, h), borderValue=0)
    dilated = cv2.dilate(shadow, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
    blurred = cv2.GaussianBlur(dilated, (101, 101), 0)
    shadow_float = blurred.astype(np.float32) / 255.0
    shadow_scaled = (shadow_float * 255 * intensity).clip(0, 255).astype(np.uint8)
    shadow_scaled = sharpen_shadow(shadow_scaled, intensity=0.3, radius=5)
    return shadow_scaled

def blend_object(background_path, object_path, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    bg = cv2.imread(background_path)
    obj = cv2.imread(object_path, cv2.IMREAD_UNCHANGED)
    if obj is None or bg is None:
        print("Error loading images")
        return
    if obj.shape[2] == 4:
        obj_rgb = obj[:, :, :3]
        alpha = obj[:, :, 3]
        _, object_mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
    else:
        obj_rgb = obj
        gray = cv2.cvtColor(obj_rgb, cv2.COLOR_BGR2GRAY)
        _, object_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        alpha = object_mask
    scale_factor = 2.0
    obj_rgb = cv2.resize(obj_rgb, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    alpha = cv2.resize(alpha, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    _, object_mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
    light_dir = detect_light_direction(bg, visualize=True)
    shadow_mask = create_realistic_shadow(object_mask, light_dir, intensity=0.55)
    bg_h, bg_w = bg.shape[:2]
    obj_h, obj_w = obj_rgb.shape[:2]
    y_pos = int((bg_h * 0.5) + (bg_h * 0.25) - (obj_h // 2) + (bg_h * 0.10) - (bg_h * 0.05))
    x_pos = (bg_w - obj_w) // 2
    x_start = max(x_pos, 0)
    y_start = max(y_pos, 0)
    x_end = min(x_start + obj_w, bg_w)
    y_end = min(y_start + obj_h, bg_h)
    obj_w_cropped = x_end - x_start
    obj_h_cropped = y_end - y_start
    obj_rgb = obj_rgb[:obj_h_cropped, :obj_w_cropped]
    alpha = alpha[:obj_h_cropped, :obj_w_cropped]
    object_mask = object_mask[:obj_h_cropped, :obj_w_cropped]
    shadow_mask = shadow_mask[:obj_h_cropped, :obj_w_cropped]
    shadow_rgb = np.zeros_like(bg)
    shadow_roi = shadow_rgb[y_start:y_end, x_start:x_end]
    shadow_mask_3ch = cv2.merge([shadow_mask] * 3)
    shadow_roi[:] = np.where(shadow_mask_3ch > 0, shadow_mask_3ch, shadow_roi)
    shadow_area = cv2.subtract(bg, shadow_rgb)
    bg = np.where(shadow_rgb > 0, shadow_area, bg)
    if obj.shape[2] == 4:
        alpha_normalized = alpha[..., None] / 255.0
        roi = bg[y_start:y_end, x_start:x_end]
        bg[y_start:y_end, x_start:x_end] = (
            roi * (1 - alpha_normalized) + obj_rgb * alpha_normalized
        ).astype(np.uint8)
    else:
        bg[y_start:y_end, x_start:x_end] = obj_rgb
    cv2.imwrite(output_path, bg)
    print(f"Saved final composition with realistic shadow at: {output_path}")

blend_object(
    background_path="park.png",
    object_path="output.png",
    output_path="Final_park.jpg"
)
