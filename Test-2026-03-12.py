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
# 创建结果目录
############################################

save_dir = "visualize_result"
os.makedirs(save_dir, exist_ok=True)


############################################
# 加载 YOLOv5-face
############################################

device = select_device('')
weights = "weights/yolov5s-face.pt"

model = attempt_load(weights, map_location=device)
stride = int(model.stride.max())
imgsz = check_img_size(640)

model.eval()


############################################
# 加载 dlib
############################################

predictor_path = "weights/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)


############################################
# 遍历图片
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
    # YOLOv5-face 推理
    ############################################

    img = letterbox(img0, imgsz)[0]

    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]

    pred = non_max_suppression_face(
        pred,
        conf_thres=0.5,
        iou_thres=0.5
    )


    ############################################
    # 是否检测到人脸
    ############################################

    if pred[0] is None or len(pred[0]) == 0:
        print("未检测到人脸")
        continue

    det = pred[0]

    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()


    ############################################
    # 选择最大人脸
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
    # 画人脸框
    ############################################

    cv2.rectangle(img0, (x1, y1), (x2, y2), (0,255,0), 2)


    ############################################
    # dlib 关键点
    ############################################

    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

    rect = dlib.rectangle(x1, y1, x2, y2)

    shape = predictor(gray, rect)

    landmarks = np.zeros((68,2), dtype="int")

    for j in range(68):
        landmarks[j] = (shape.part(j).x, shape.part(j).y)


    ############################################
    # 画68关键点
    ############################################

    for (x,y) in landmarks:
        cv2.circle(img0, (x,y), 2, (0,0,255), -1)


    ############################################
    # 保存结果
    ############################################

    save_path = os.path.join(save_dir, f"result_{i:02d}.jpg")

    cv2.imwrite(save_path, img0)

    print(f"结果保存: {save_path}")


    ############################################
    # 尝试显示
    ############################################

    try:

        cv2.imshow("Face + 68 Landmarks", img0)
        print("按任意键查看下一张")

        cv2.waitKey(0)

    except:
        print("当前环境不支持imshow，已保存图片")


cv2.destroyAllWindows()