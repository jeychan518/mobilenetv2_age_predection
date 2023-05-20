import os
import cv2
import numpy as np
import onnxruntime as ort

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def preprocess_cls_image(img, img_sz):
    h, w = img.shape[:2]
    if h > img_sz or w > img_sz:
        if h > w:
            new_h = img_sz
            new_w = int(w * img_sz / h)
        else:
            new_w = img_sz
            new_h = int(h * img_sz / w)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 计算需要添加的上下边框的像素数
    border_h = int((img_sz - img.shape[0]) / 2)
    # 计算需要添加的左右边框的像素数
    border_w = int((img_sz - img.shape[1]) / 2)

    # 使用cv2.copyMakeBorder()函数为图像添加边框
    img = cv2.copyMakeBorder(img, border_h, border_h, border_w, border_w, cv2.BORDER_CONSTANT,
                             value=(127, 127, 127))
    img = cv2.resize(img, (img_sz, img_sz), interpolation=cv2.INTER_LINEAR)

    im = np.float32(img)
    im /= 255.0
    im -= mean
    im /= std
    im = im.transpose(2, 0, 1)
    batch_data = np.expand_dims(im, axis=0)
    return batch_data


if __name__ == '__main__':
    session = ort.InferenceSession(r'models/model.onnx', providers=['CPUExecutionProvider'])
    for img_file in os.listdir('./images'):
        img = cv2.imread(os.path.join('./images', img_file))
        input = preprocess_cls_image(img, 128)
        age = session.run(None, input_feed={'input.1': input})[0][0][0] * 100
        print(img_file, age)
        cv2.putText(img, f'age:{int(age)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imwrite(os.path.join('./output', img_file), img)
