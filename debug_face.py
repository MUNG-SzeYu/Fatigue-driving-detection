import cv2
import dlib

# 初始化
detector = dlib.get_frontal_face_detector()

image_path = "data/driver_01.jpg"

image = cv2.imread(image_path)

if image is None:
    print("图片读取失败")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 第二个参数是上采样次数
faces = detector(gray, 3)

print(f"检测到人脸数量: {len(faces)}")

for i, face in enumerate(faces):
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()