import os
import cv2
import torch
import numpy as np
import dlib

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords
from utils.torch_utils import select_device


############################################
# EAR / MAR 计算函数
############################################

def euclidean_dist(p1, p2):
    return np.linalg.norm(p1 - p2)


def compute_EAR(eye):
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])
    C = euclidean_dist(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def compute_MAR(mouth):
    A = euclidean_dist(mouth[2], mouth[10])
    B = euclidean_dist(mouth[3], mouth[9])
    C = euclidean_dist(mouth[4], mouth[8])
    D = euclidean_dist(mouth[0], mouth[6])
    return (A + B + C) / (2.0 * D)


############################################
# 1 加载 YOLOv5-face
############################################

device = select_device('')
weights = "weights/yolov5s-face.pt"

model = attempt_load(weights, map_location=device)
stride = int(model.stride.max())
imgsz = check_img_size(640)

model.eval()


############################################
# 2 加载 dlib landmark
############################################

predictor_path = "weights/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)


############################################
# 3 遍历图片
############################################

for i in range(1, 21):

    img_path = f"data/google/driver_{i:02d}.jpg"

    if not os.path.exists(img_path):
        print(f"{img_path} 不存在")
        continue

    print("\n=================================")
    print(f"处理图片: {img_path}")

    img0 = cv2.imread(img_path)

    if img0 is None:
        print("图片读取失败")
        continue


    ############################################
    # 4 YOLOv5-face 推理
    ############################################

    img = letterbox(img0, imgsz)[0]

    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]

    pred = non_max_suppression_face(
        pred,
        conf_thres=0.5,
        iou_thres=0.5
    )


    ############################################
    # 5 是否检测到人脸
    ############################################

    if pred[0] is None or len(pred[0]) == 0:
        print("未检测到人脸")
        continue

    print("检测到人脸")

    det = pred[0]

    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()


    ############################################
    # 6 选择最大人脸
    ############################################

    max_area = 0
    best_box = None

    for det_item in det:

        x1 = int(det_item[0])
        y1 = int(det_item[1])
        x2 = int(det_item[2])
        y2 = int(det_item[3])

        area = (x2 - x1) * (y2 - y1)

        if area > max_area:
            max_area = area
            best_box = (x1, y1, x2, y2)

    if best_box is None:
        print("人脸检测失败")
        continue

    x1, y1, x2, y2 = best_box


    ############################################
    # 7 dlib 关键点检测
    ############################################

    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

    rect = dlib.rectangle(x1, y1, x2, y2)

    shape = predictor(gray, rect)

    landmarks = np.zeros((68, 2), dtype="int")

    for j in range(68):
        landmarks[j] = (shape.part(j).x, shape.part(j).y)


    ############################################
    # 8 计算 EAR
    ############################################

    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]

    EAR_left = compute_EAR(left_eye)
    EAR_right = compute_EAR(right_eye)

    EAR = (EAR_left + EAR_right) / 2.0


    ############################################
    # 9 计算 MAR
    ############################################

    mouth = landmarks[48:68]

    MAR = compute_MAR(mouth)


    ############################################
    # 10 输出结果
    ############################################

    print(f"EAR: {EAR:.4f}")
    print(f"MAR: {MAR:.4f}")