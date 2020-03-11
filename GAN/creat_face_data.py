import os
from PIL import Image
import numpy as np
import traceback

anno_src = r"G:\celeba\Anno\new_celeba_label.txt"
img_dir = r"G:\celeba\img_celeba"



save_path = r"F:"

for face_size in [96]:#12,24,48

    print("gen %i image" % face_size)
    # 样本图片存储路径
    positive_image_dir = os.path.join(save_path,"faces")

    for dir_path in [positive_image_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    positive_count = 0
    for i, line in enumerate(open(anno_src)):
        if i < 2:
            continue
        try:
            strs = line.strip().split(" ")
            strs = list(filter(bool, strs))
            image_filename = strs[0].strip()
            print(image_filename)
            image_file = os.path.join(img_dir, image_filename)

            with Image.open(image_file) as img:
                img_w, img_h = img.size
                x1 = float(strs[1].strip())
                y1 = float(strs[2].strip())
                w = float(strs[3].strip())
                h = float(strs[4].strip())
                x2 = float(x1 + w)
                y2 = float(y1 + h)



                if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                    continue
                boxes = [x1, y1, x2, y2]

                # 剪切下图片，并进行大小缩放
                face_crop = img.crop(boxes)
                face_resize = face_crop.resize((face_size, face_size))

                face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                positive_count += 1
            if positive_count==50000:
                break
        except Exception as e:
                traceback.print_exc()
