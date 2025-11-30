# vision/inspection.py
import cv2
from utils.logger import get_logger
from shared.state import state

logger = get_logger("Inspection")


def inspect_target(frame, target_box, margin=50):
    """
    使用 state.inspections 中的检测结果进行缺陷判断，不重新调用检测器。
    返回: (is_defective, defect_list, roi_vis_frame)
    """
    h, w = frame.shape[:2]
    (x1, y1), (x2, y2) = target_box

    # --- 5. 缺陷判断逻辑 ---
    is_defective = False
    found_defects = []

    # --- 1. 使用 state.inspections ---
    with state.lock:
        detections = state.inspections.copy()

    # --- 2. 裁剪 ROI 区域（仅用于显示，不用于检测） ---
    crop_x1 = max(0, x1 - margin)
    crop_y1 = max(0, y1 - margin)
    crop_x2 = min(w, x2 + margin)
    crop_y2 = min(h, y2 + margin)
    roi_vis = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

    # --- 3. 在 ROI 内画所有检测框 ---
    for det in detections:
        (bx1, by1), (bx2, by2) = det["bbox"]
        label = det["label"]

        # 转换到 ROI 坐标
        rx1 = max(0, bx1 - crop_x1)
        ry1 = max(0, by1 - crop_y1)
        rx2 = min(crop_x2 - crop_x1, bx2 - crop_x1)
        ry2 = min(crop_y2 - crop_y1, by2 - crop_y1)

        # 只画在 ROI 内的框
        if rx1 >= rx2 or ry1 >= ry2:
            continue

        # 颜色区分目标/缺陷（可选）
        if label == "workpiece" or label == "normal":
            color = (0, 255, 0)  # 绿色
        else:
            is_defective = True
            found_defects.append(label)
            color = (0, 0, 255)  # 红色

        cv2.rectangle(roi_vis, (rx1, ry1), (rx2, ry2), color, 2)
        cv2.putText(roi_vis, label, (rx1, ry1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # --- 4. 寻找全图中的最大 workpiece 框 ---
    workpiece_box = None
    max_area = 0
    for det in detections:
        if det["label"] == "workpiece":
            (bx1, by1), (bx2, by2) = det["bbox"]
            area = (bx2 - bx1) * (by2 - by1)
            if area > max_area:
                max_area = area
                workpiece_box = det["bbox"]

    # for det in detections:
    #     label = det["label"]
    #     if label in ["loose", "dirty"]:
    #         if workpiece_box is None:
    #             is_defective = True
    #             found_defects.append(label)
    #         else:
    #             bias = 10
    #             (dx1, dy1), (dx2, dy2) = det["bbox"]
    #             d_cx = (dx1 + dx2) / 2
    #             d_cy = (dy1 + dy2) / 2
    #             (wx1, wy1), (wx2, wy2) = workpiece_box
    #
    #             if wx1 - bias <= d_cx <= wx2 + bias and wy1 - bias <= d_cy <= wy2 + bias:
    #                 is_defective = True
    #                 found_defects.append(label)
    #                 logger.info(f"Defect '{label}' found inside workpiece.")
    #             else:
    #                 logger.info(f"Defect '{label}' detected but outside workpiece bounds. Ignored.")

    return is_defective, found_defects, roi_vis
