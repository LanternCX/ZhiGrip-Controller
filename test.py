from ultralytics import YOLO
import cv2
import numpy as np
import os
import torch


class YOLODetector:
    def __init__(self, model_path, camera_index=0):
        """
        初始化YOLO检测器

        Args:
            model_path (str): 模型文件路径
            camera_index (int): 摄像头索引，默认为0
        """
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 加载模型
        self.model = YOLO(model_path)
        self.camera_index = camera_index

        # 获取模型的类别名称（如果有）
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}

    def detect_frame(self, frame, conf_threshold=0.5):
        """
        对单帧图像进行目标检测

        Args:
            frame: 图像帧
            conf_threshold (float): 置信度阈值

        Returns:
            results: 检测结果
        """
        results = self.model(frame, conf=conf_threshold)
        return results

    def initialize_camera(self):
        """
        初始化摄像头参数

        Returns:
            cap: 摄像头对象
        """
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError("无法打开摄像头，请检查连接！")

        # 设置摄像头参数 - 提高曝光度
        cap.set(cv2.CAP_PROP_EXPOSURE, -1)  # 大幅提高曝光度（原为15）
        cap.set(cv2.CAP_PROP_EXPOSURE, -1)  # 大幅提高曝光度（原为15）
        cap.set(cv2.CAP_PROP_BRIGHTNESS, -2)  # 亮度
        # cap.set(cv2.CAP_PROP_CONTRAST, 64)      # 对比度
        # cap.set(cv2.CAP_PROP_SATURATION, 64)    # 饱和度
        # cap.set(cv2.CAP_PROP_SHARPNESS, 3)      # 锐度
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # 关闭自动对焦

        return cap

    def get_sharpness_measure(self, frame):
        """
        计算图像清晰度指标（拉普拉斯方差法）

        Args:
            frame: 图像帧

        Returns:
            float: 清晰度数值
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var

    def draw_detections(self, frame, results):
        """
        在图像上绘制检测结果（不显示位置信息）

        Args:
            frame: 图像帧
            results: 检测结果

        Returns:
            frame: 带有标注的图像帧
        """
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

                    # 获取类别和置信度
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # 根据类别绘制不同颜色的框
                    color = self._get_class_color(cls)
                    class_name = self.class_names.get(cls, f"Class {cls}")
                    label = f"{class_name}: {conf:.2f}"

                    # 绘制边界框和标签（不显示中心点和坐标信息）
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def _get_class_color(self, class_id):
        """
        根据类别ID返回对应颜色

        Args:
            class_id (int): 类别ID

        Returns:
            tuple: BGR颜色值
        """
        # 定义一些默认颜色
        colors = [
            (255, 0, 0),  # 蓝色
            (0, 0, 255),  # 红色
            (0, 255, 0),  # 绿色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 紫色
            (0, 255, 255),  # 黄色
        ]

        return colors[class_id % len(colors)]

    def run_realtime_detection(self, conf_threshold=0.6):
        """
        运行实时检测（不返回位置信息）

        Args:
            conf_threshold (float): 置信度阈值
        """
        # 初始化摄像头
        cap = self.initialize_camera()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法获取画面，退出程序")
                    break

                # 计算当前图像清晰度
                sharpness = self.get_sharpness_measure(frame)

                # 使用模型进行检测
                results = self.detect_frame(frame, conf_threshold)

                # 绘制检测结果（不显示位置信息）
                frame = self.draw_detections(frame, results)

                # 在画面上显示清晰度数值
                cv2.putText(frame, f"Sharpness: {sharpness:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 显示处理结果
                cv2.imshow("YOLO实时检测", frame)

                # 按键控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        finally:
            # 释放资源
            cap.release()
            cv2.destroyAllWindows()


def main():
    """
    主函数 - 演示如何使用YOLODetector类
    """
    # 模型路径
    model_path = r"./data/best.pt"

    try:
        # 创建检测器实例
        detector = YOLODetector(model_path, camera_index=0)

        # 显示模型信息
        print(f"模型类别: {detector.class_names}")

        # 运行实时检测
        print("开始实时检测，按 'q' 键退出...")
        detector.run_realtime_detection(conf_threshold=0.6)

    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()