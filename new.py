#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Senray Inc. All Rights Reserved.
# WU KAI Programmed
# Version 20250806
#

from ultralytics import YOLO
import os
import socket
import shutil
import cv2


def has_at_least_8_jpg(folder_path):
    count = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jpeg'):
            count += 1
            if count >= 8:
                return True
    return False

def get_detection_box_coordinates(results):
    """
    从YOLO检测结果中获取检测框的坐标
    参数:
        results: YOLO检测结果
    返回:
        检测框的左上右下坐标 (x1, y1, x2, y2)
    """
    # 检查是否检测到对象
    if len(results[0].boxes) == 0:
        return None

    # 获取置信度最高的检测框
    best_box = results[0].boxes[0]

    # 获取边界框坐标 (xyxy格式)
    box_coords = best_box.xyxy[0].cpu().numpy()

    # 转换为整数
    x1, y1, x2, y2 = map(int, box_coords)

    return x1, y1, x2, y2


def crop_image(image, top_left, bottom_right):
    """
    使用指定坐标裁剪图像（内存中处理）
    参数:
        image: 输入图像数组 (numpy.ndarray)
        top_left: 左上角坐标 (x1, y1)
        bottom_right: 右下角坐标 (x2, y2)
    返回:
        裁剪后的图像数组
    """
    # 获取图像尺寸
    h, w = image.shape[:2]

    # 转换坐标为整数
    x1, y1 = map(int, top_left)
    x2, y2 = map(int, bottom_right)

    # 验证坐标有效性
    if x1 >= x2 or y1 >= y2:
        raise ValueError("坐标无效：左上角坐标必须小于右下角坐标")

    # 确保坐标在图像范围内
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    # 裁剪图像
    cropped = image[y1:y2, x1:x2]

    # 检查裁剪区域是否有效
    if cropped.size == 0:
        raise ValueError("裁剪区域无效")

    return cropped


def crop_with_detection_box(image, results):
    """
    使用检测框坐标裁剪图像（内存中处理）
    参数:
        image: 输入图像数组
        results: YOLO检测结果
    返回:
        裁剪后的图像数组
    """
    # 检查是否检测到对象
    if len(results[0].boxes) == 0:
        return None

    # 获取置信度最高的检测框（第一个框）
    best_box = results[0].boxes[0]

    # 获取边界框坐标 (xyxy格式)
    box_coords = best_box.xyxy[0].cpu().numpy()

    # 转换为整数
    x1, y1, x2, y2 = map(int, box_coords)

    # 扩展边界框 (左上各减50，右下各加50)
    expand_pixels = 50
    top_left = (x1 - expand_pixels, y1 - expand_pixels)
    bottom_right = (x2 + expand_pixels, y2 + expand_pixels)

    # 裁剪图像
    return crop_image(image, top_left, bottom_right)


def crop_with_fixed_box(image):
    """
    使用固定坐标裁剪图像（内存中处理）
    参数:
        image: 输入图像数组
    返回:
        裁剪后的图像数组
    """
    # 固定裁剪坐标
    top_left = (600, 700)  # (x1, y1)
    bottom_right = (1500, 2800)  # (x2, y2)

    # 裁剪图像
    return crop_image(image, top_left, bottom_right)

def delete_all_jpg(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jpeg'):
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
                print(f"已删除: {file_path}")
            except Exception as e:
                print(f"删除失败 {file_path}: {e}")


def move_and_clean_predict_folder(output_folder):
    """将predict文件夹中的图片移动到目标文件夹并删除predict文件夹"""
    predict_folder = os.path.join(output_folder, 'predict')

    if os.path.exists(predict_folder):
        # 移动所有图片到目标文件夹
        for filename in os.listdir(predict_folder):
            if filename.lower().endswith('.jpeg'):
                src = os.path.join(predict_folder, filename)
                dst = os.path.join(output_folder, filename)

                # 如果目标文件已存在，先删除
                if os.path.exists(dst):
                    os.remove(dst)

                shutil.move(src, dst)
                print(f"已移动图片: {filename}")

        # 删除predict文件夹
        try:
            shutil.rmtree(predict_folder)
            print(f"已删除predict文件夹: {predict_folder}")
        except Exception as e:
            print(f"删除predict文件夹失败: {e}")


# 创建socket连接
socket_client = socket.socket()
socket_client.connect(('localhost', 8888))

input_folder = r"D:\newproj\待处理图片"
result_file = r"D:\newproj\result\result.txt"
output_folder = r"D:\newproj\处理后图片"

# 确保输出目录存在
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.dirname(result_file), exist_ok=True)
# 支持的图片扩展名
valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
model = YOLO(model='D:\模型检测\pos.pt')
model2 = YOLO(model='D:\模型检测\ys85.pt')

while True:
    recv_data = socket_client.recv(1024).decode("UTF-8")
    if recv_data == 'picjet':
        if has_at_least_8_jpg(input_folder):
            image_files = []
            for file in os.listdir(input_folder):
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    image_files.append(os.path.join(input_folder, file))
            result_bits = []
            for img_path in image_files:
                try:
                    filename = os.path.basename(img_path)
                    results = model.predict(source=img_path,save=False)
                    box_coords = get_detection_box_coordinates(results)
                    original_image = cv2.imread(img_path)
                    if box_coords is not None:
                        cropped_image = crop_with_detection_box(original_image, results)
                    else:
                        cropped_image = crop_with_fixed_box(original_image)

                    results2 = model2.predict(source=cropped_image,save=False)
                    for i in results2:
                        img = i.orig_img.copy()  # 复制原始图像

                        # 绘制检测结果（红色框 + 大字体）
                        for box in i.boxes:
                            xyxy = box.xyxy[0].tolist()
                            class_id = int(box.cls[0])
                            conf = float(box.conf[0])

                            # 绘制红色边界框 (BGR: 0,0,255)
                            cv2.rectangle(img,
                                          (int(xyxy[0]), int(xyxy[1])),
                                          (int(xyxy[2]), int(xyxy[3])),
                                          (0, 0, 255), 7)

                            # 准备标签文本
                            label = f"{i.names[class_id]} {conf:.2f}"

                            # 设置大字体
                            font_scale = 3.5
                            thickness = 2
                            font = cv2.FONT_HERSHEY_SIMPLEX

                            # 计算文本大小并绘制背景
                            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                            cv2.rectangle(img,
                                      (int(xyxy[0]), int(xyxy[1]) - text_height - 10),
                                      (int(xyxy[0]) + text_width, int(xyxy[1])),
                                      (0, 0, 255), -1)

                            # 绘制白色文本
                            cv2.putText(img, label,
                                    (int(xyxy[0]), int(xyxy[1]) - 5),
                                    font, font_scale, (255, 255, 255), thickness)
                        output_path = os.path.join(output_folder, filename)
                        cv2.imwrite(output_path, img)
                    # 生成结果字符串 (1表示有缺陷，0表示无缺陷)

                    for res in results2:
                        if len(res.boxes) > 0:
                            result_bits.append("1")
                        else:
                            result_bits.append("0")
                except Exception as e:
                    print(f"处理失败: {os.path.basename(img_path)} - {str(e)}")
            # 确保有8个结果
            while len(result_bits) < 8:
                result_bits.append("0")

            result_str = " ".join(result_bits[:8])

            # 写入结果文件
            with open(result_file, 'w', encoding='utf-8') as f:
                    f.write(result_str)

            # 发送处理结果
            if "1" in result_str:
                socket_client.send("ng".encode("UTF-8"))
            else:
                socket_client.send("ok".encode("UTF-8"))

            # 删除原始图片
            delete_all_jpg(input_folder)


