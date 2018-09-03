"""
@author: yi
time:Sept 2nd,2018
task：identify captcha
future work:add more kind of captcha for training set and optimizing the identification algorithm.
"""
# coding=utf-8
import glob
import sys
import os
import time
import numpy as np
import cv2

TRAIN_DIR = "train"
TEST_DIR = "test"
CHAR_DIR = "char"
LABEL_DIR = "label"


def get_rect_box(contours):
    """
    convert contours to boxes
    each box is a rectangle consisting of 4 points
    if there is connected characters, split the contour

    param:
        contours -- 字符轮廓

    return:
        result -- 一张图片中4个字符的外接矩形框
    """
    ws = []
    valid_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 7:
            continue
        valid_contours.append(contour)
        ws.append(w)

    w_min = min(ws)
    w_max = max(ws)

    result = []
    if len(valid_contours) == 4:
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            box = np.int0([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            result.append(box)
    elif len(valid_contours) == 3:
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w == w_max:
                box_left = np.int0(
                    [[x, y], [x + w / 2, y], [x + w / 2, y + h], [x, y + h]])
                box_right = np.int0(
                    [[x + w / 2, y], [x + w, y], [x + w, y + h], [x + w / 2, y + h]])
                result.append(box_left)
                result.append(box_right)
            else:
                box = np.int0([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
                result.append(box)
    elif len(valid_contours) == 2:
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w == w_max and w_max >= w_min * 2:
                box_left = np.int0(
                    [[x, y], [x + w / 3, y], [x + w / 3, y + h], [x, y + h]])
                box_mid = np.int0(
                    [[x + w / 3, y], [x + w * 2 / 3, y], [x + w * 2 / 3, y + h], [x + w / 3, y + h]])
                box_right = np.int0(
                    [[x + w * 2 / 3, y], [x + w, y], [x + w, y + h], [x + w * 2 / 3, y + h]])
                result.append(box_left)
                result.append(box_mid)
                result.append(box_right)
            elif w_max < w_min * 2:
                box_left = np.int0(
                    [[x, y], [x + w / 2, y], [x + w / 2, y + h], [x, y + h]])
                box_right = np.int0(
                    [[x + w / 2, y], [x + w, y], [x + w, y + h], [x + w / 2, y + h]])
                result.append(box_left)
                result.append(box_right)
            else:
                box = np.int0([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
                result.append(box)
    elif len(valid_contours) == 1:
        contour = valid_contours[0]
        x, y, w, h = cv2.boundingRect(contour)
        box0 = np.int0(
            [[x, y], [x + w / 4, y], [x + w / 4, y + h], [x, y + h]])
        box1 = np.int0([[x + w / 4, y], [x + w * 2 / 4, y],
                        [x + w * 2 / 4, y + h], [x + w / 4, y + h]])
        box2 = np.int0([[x + w * 2 / 4, y], [x + w * 3 / 4, y],
                        [x + w * 3 / 4, y + h], [x + w * 2 / 4, y + h]])
        box3 = np.int0([[x + w * 3 / 4, y], [x + w, y],
                        [x + w, y + h], [x + w * 3 / 4, y + h]])
        result.extend([box0, box1, box2, box3])
    elif len(valid_contours) > 4:
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            box = np.int0([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            result.append(box)
    result = sorted(result, key=lambda x: x[0][0])
    return result


def process_im(im):
    """
    process image including denoise(去噪), thresholding

    param:
        im -- 图像像素值数组

    return:
        im_res -- 处理后的图像像素值数组,结果是灰度且二值化的像素值数组
    """
    # gray processing
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # thresholding
    ret, im_inv = cv2.threshold(im_gray, 127, 255, cv2.THRESH_BINARY_INV)
    # denoise
    kernel = 1 / 16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    im_blur = cv2.filter2D(im_inv, -1, kernel)
    # thresholding
    ret, im_res = cv2.threshold(im_blur, 127, 255, cv2.THRESH_BINARY)
    return im_res


def split_code(filepath):
    """
    split a captcha image into single character and restore them to char folder.

    param:
        filepath --  path of file

    return:
        None
    """

    filename = filepath.split("/")[-1]
    filename_ts = filename.split(".")[0]
    im = cv2.imread(filepath)
    im_res = process_im(im)

    # 根据二值化后的黑白图像很容易获取前景的轮廓，使用findcontours
    im2, contours, hierarchy = cv2.findContours(
        im_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = get_rect_box(contours)
    if len(boxes) != 4:
        print(filepath)

    if not os.path.exists(CHAR_DIR):
        os.makedirs(CHAR_DIR)

    for box in boxes:
        cv2.drawContours(im, [box], 0, (0, 0, 255), 2)
        roi = im_res[box[0][1]:box[3][1], box[0][0]:box[1][0]]
        roistd = cv2.resize(roi, (30, 30))
        timestamp = int(time.time() * 1e6)
        filename = "{}.jpg".format(timestamp)
        filepath = os.path.join("char", filename)
        cv2.imwrite(filepath, roistd)

    # cv2.imshow("image", im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def split_all():
    """
    split all captcha images into single character in training set, and restore them into char folder.

    return:
        None
    """

    files = os.listdir(TRAIN_DIR)
    for filename in files:
        filename_ts = filename.split(".")[0]
        patt = "label/{}_*".format(filename_ts)
        saved_chars = glob.glob(patt)
        if len(saved_chars) == 4:
            print("{} done".format(filepath))
            continue
        filepath = os.path.join(TRAIN_DIR, filename)
        split_code(filepath)


def label_data():
    """
    quickly label data for training set by function code rather than renaming for label files

    return:
        None
    """
    if not os.path.exists(LABEL_DIR):
        os.makedirs(LABEL_DIR)

    files = os.listdir("char")
    for filename in files:
        filename_ts = filename.split(".")[0]
        # 判断当前字符图片文件是否已经存在于label文件夹中
        # 如果存在则循环下一张图，否则开始打标签
        patt = "label/{}_*".format(filename_ts)
        saved_num = len(glob.glob(patt))
        if saved_num == 1:
            print("{} done".format(patt))
            continue

        filepath = os.path.join("char", filename)
        im = cv2.imread(filepath)
        cv2.imshow("image", im)
        key = cv2.waitKey(0)

        # if key is ESC,and then exit program
        if key == 27:
            sys.exit()

        # if key is Backspace,and then skip this character
        if key == 13:
            continue
        char = chr(key)
        filename_ts = filename.split(".")[0]
        outfile = "{}_{}.jpg".format(filename_ts, char)
        outpath = os.path.join("label", outfile)
        cv2.imwrite(outpath, im)


def analyze_label():
    files = os.listdir("label")
    label_count = {}
    for filename in files:
        label = filename.split(".")[0].split("_")[1]
        label_count.setdefault(label, 0)
        label_count[label] += 1
    print(label_count)


def load_data():
    """
    load all data for training set,including training set data and labels

    return:
        [samples, label_ids, id_label_map] -- data set,list
        samples -- all single characters's pixel value matrix in training set of shape (804,900)
        label_ids -- subscript id vector of shape (804,1) and the every id value corresponds to real label
        id_label_map -- the map dictionary of subscript id to real label
    """
    filenames = os.listdir("label")
    samples = np.empty((0, 900))
    labels = []
    for filename in filenames:
        filepath = os.path.join("label", filename)
        label = filename.split(".")[0].split("_")[-1]
        labels.append(label)
        im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        sample = im.reshape((1, 900)).astype(np.float32)
        samples = np.append(samples, sample, 0)
    samples = samples.astype(np.float32)
    unique_labels = list(set(labels))
    unique_ids = list(range(len(unique_labels)))
    label_id_map = dict(zip(unique_labels, unique_ids))
    id_label_map = dict(zip(unique_ids, unique_labels))
    label_ids = list(map(lambda x: label_id_map[x], labels))
    label_ids = np.array(label_ids).reshape((-1, 1)).astype(np.float32)
    return [samples, label_ids, id_label_map]


def get_code(file_path):
    """
    identify a new captcha image

    param:
        file_path -- a new captcha image's path,String.

    return:
        result -- the result of identification,list.
    """
    # training
    [samples, label_ids, id_label_map] = load_data()
    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, label_ids)

    # prediction
    im = cv2.imread(file_path)
    im_res = process_im(im)
    im2, contours, hierarchy = cv2.findContours(
        im_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = get_rect_box(contours)
    if len(boxes) != 4:
        print("cannot get code")

    result = []
    for box in boxes:
        roi = im_res[box[0][1]:box[3][1], box[0][0]:box[1][0]]
        roistd = cv2.resize(roi, (30, 30))
        sample = roistd.reshape((1, 900)).astype(np.float32)
        ret, results, neighbours, distances = model.findNearest(sample, k=3)
        label_id = int(results[0, 0])
        label = id_label_map[label_id]
        result.append(label)
    return result


def test_data():
    """
    test captcha images in test set

    return:
        None
    """
    test_files = os.listdir("test")
    total = 0
    correct = 0
    for filename in test_files:
        filepath = os.path.join("test", filename)
        preds = get_code(filepath)
        chars = filename.split(".")[0]
        print(chars, preds)
        for i in range(len(chars)):
            if chars[i] == preds[i]:
                correct += 1
            total += 1
    print(correct / total)


if __name__ == "__main__":
    file_path = r"C:\Users\yiguanxian\Desktop\1535348797742994.jpg"
    im = cv2.imread(file_path)
    cv2.imshow('captcha', im)
    cv2.waitKey(30)
    result = get_code(file_path)
    print(result)
