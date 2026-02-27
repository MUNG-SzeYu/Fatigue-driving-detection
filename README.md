# Fatigue-driving-detection
Fatigue driving detection (for learning)

这是我的疲劳驾驶检测模型，用于个人学习以及vibe coding。
*更新于2/27/2026*

# 数据预处理
dataProcess.py是对data文件夹中的数据进行处理的python程序，现有data文件夹中的图片来源于google

# 特征提取
main.py是用于计算EAR和MAR的程序，使用Dlib库对图片中的人脸进行提取并计算EAR和MAR并输出，若检测不出人脸会进行报告。