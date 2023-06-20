import os
import shutil

box_img_dir = "squat/"
origin_img_dir = "squat_origin/images/"

box_img_dir_image = "images/"
if not os.path.exists(box_img_dir+box_img_dir_image):
    os.makedirs(box_img_dir+box_img_dir_image)

if __name__ == "__main__":
    box_imgs = os.listdir(box_img_dir)
    count = 0
    for img in os.listdir(origin_img_dir):
        if img.endswith(".jpg"):
            if img in box_imgs:
                print(img)
                count += 1
                shutil.copy(origin_img_dir+img, box_img_dir+box_img_dir_image)
        else:
            continue
    print("origin img got: ", len(os.listdir(origin_img_dir)))
    print("transfer: ", count)