import torch.nn as nn 
import os
import cv2
import dlib
import numpy as np
from scipy.spatial import distance

# ========== 初始化 ==========
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# ========== EAR ==========
def compute_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)
# EAR小于0.2通常被认为是闭眼状态，可以根据需要调整这个阈值

# ========== MAR ==========
def compute_MAR(mouth):
    A = distance.euclidean(mouth[1], mouth[9])
    B = distance.euclidean(mouth[3], mouth[7])
    C = distance.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)
# MAR大于0.5通常被认为是张嘴状态，可以根据需要调整这个阈值

# ========== 提取特征 ==========
def extract_features(image_path):
    image = cv2.imread(image_path)  # 返回图片
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图像

    faces = detector(gray) # 检测人脸

    if len(faces) == 0:
        print(f"{image_path} 未检测到人脸")
        return None

    face = faces[0] #提取出第一张人脸
    
    shape = predictor(gray, face)

    landmarks = np.zeros((68, 2))

    for i in range(68):
        landmarks[i] = (shape.part(i).x, shape.part(i).y)

    # 眼睛
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    EAR = (compute_EAR(left_eye) + compute_EAR(right_eye)) / 2.0

    # 嘴巴
    mouth = landmarks[48:68]
    MAR = compute_MAR(mouth)

    return EAR, MAR

# ========== 主程序 ==========
data_folder = "data"

print("\n开始处理图片...\n")

for filename in os.listdir(data_folder):
    if filename.endswith(".jpg"):
        path = os.path.join(data_folder, filename)
        result = extract_features(path)

        if result is not None:
            EAR, MAR = result
            print(f"{filename}  ->  EAR: {EAR:.3f}  MAR: {MAR:.3f}")

print("\n处理完成。")