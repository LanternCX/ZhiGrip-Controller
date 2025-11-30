from .base import BaseDetector
from utils.logger import get_logger
import cv2
import os
from ultralytics import YOLO
from shared.state import state  # 导入共享状态

logger = get_logger("YoloDetector")

class YoloDetector(BaseDetector):
    """
    YOLOv11 模型检测器（离线加载本地权重）
    """
    def __init__(self, tag=None, model_path=None, conf_thres=0.50, filter_by_tag=True):
        super().__init__(tag)
        if model_path is None:
            current_dir = os.path.dirname(__file__)
            model_path = os.path.abspath(os.path.join(current_dir, "../../data/best.pt"))
        self.model_path = model_path
        self.model = None
        self.conf_thres = conf_thres
        self.filter_by_tag = filter_by_tag
        self._model_loaded = False  # 保证只加载一次

    def _load_model(self):
        if not self._model_loaded:
            if not os.path.exists(self.model_path):
                logger.error(f"[YOLO] Model file not found: {self.model_path}")
                return
            logger.info(f"[YOLO] Loading model from {self.model_path}")
            self.model = YOLO(self.model_path, verbose=False)
            self._model_loaded = True

    def detect(self, frame):
        self._load_model()
        if self.model is None:
            return [], frame

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(img, conf=self.conf_thres, verbose=False)

        boxes = []

        # --- 线程安全写入 state.inspections ---
        with state.lock:
            state.inspections.clear()  # 清空上一次检测结果

            for result in results:
                for det in result.boxes:
                    x1, y1, x2, y2 = map(int, det.xyxy[0])
                    conf = float(det.conf[0])
                    cls_id = int(det.cls[0])
                    label = self.model.names[cls_id]

                    if conf < self.conf_thres:
                        continue

                    # 所有标签都加入 inspections
                    state.inspections.append({
                        "label": label,
                        "conf": conf,
                        "bbox": ((x1, y1), (x2, y2))
                    })

                    # 画框和 boxes 根据 filter_by_tag 控制
                    if self.filter_by_tag and self.tag is not None and label != self.tag:
                        continue

                    boxes.append(((x1, y1), (x2, y2)))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        logger.debug(f"[YOLO] Detected {len(boxes)} boxes (tag='{self.tag}'). Total inspections: {len(state.inspections)}")
        return boxes, frame
