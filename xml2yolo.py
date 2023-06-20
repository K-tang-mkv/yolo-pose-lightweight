import os
import xml.etree.ElementTree as ET
import cv2

# 定义xml文件路径
ROOT = "squat/"
xml_dir = "squat_origin/info/"
img_dir = "squat/images/"
label_dir = "labels/"

# os.makedirs(ROOT+label_dir, exist_ok=True)

if __name__ == "__main__":
    count = 0
    invalid_count = 0
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith(".xml"):
            # 解析xml文件
            tree = ET.parse(xml_dir+xml_file)
            root = tree.getroot()

            # 获取图片名称和宽高信息
            image_name = xml_file[:-6] + ".jpg"
            if image_name not in os.listdir(img_dir):
                continue
            if f"{image_name[:-4]}.txt" not in os.listdir(ROOT+label_dir):
                continue
            print(xml_file)
            count += 1
            img = cv2.imread(img_dir+image_name)
            image_height, image_width, _ = img.shape

            # 遍历关键点信息
            keypoints = root.find('keypoints')
            bbox = []
            for kp in keypoints.findall('keypoint'):
                kp_name = kp.get('name')
                kp_x = float(kp.get('x'))
                kp_y = float(kp.get('y'))
                kp_visible = int(kp.get('visible'))

                # 计算归一化的x和y
                norm_x = kp_x / image_width
                norm_y = kp_y / image_height

                # 如果x和y都为0，则visible设为0
                if kp_x == 0 and kp_y == 0:
                    kp_visible = 0

                bbox.append((norm_x, norm_y, kp_visible))

            # 将bbox转换为yolo格式
            yolo_bbox = []
            for x, y, visible in bbox:
                yolo_bbox.append(f"{x:.6f} {y:.6f} {visible:.6f}")

            with open(f"{ROOT+label_dir+image_name[:-4]}.txt", "r") as f:
                lines = f.readlines()
            if len(lines) == 1:
                # 将yolo格式的bbox写入txt文件
                with open(f"{ROOT+label_dir+image_name[:-4]}.txt", "r+") as f:
                    content = f.read().rstrip()
                    f.seek(0,0)
                    f.write(content + " "  + " ".join(yolo_bbox))
            else:
                os.remove(f"{ROOT+label_dir+image_name[:-4]}.txt")
                os.remove(f"{ROOT + 'images/' + image_name}")
                print(len(lines))
                invalid_count += 1
    print("invalid count", invalid_count)
    print(count)