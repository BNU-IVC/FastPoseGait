import cv2 as cv
import numpy as np
import pickle
import copy
import math
import imageio
import argparse

'''
Script Example:
    python vis_out.py --pkl_path=/Dataset/CASIA-B/001/bg-01/000/000.pkl --out_dir=./vis_dir --out_gif_path=./vis_input.gif
This file is to visualize the input skeletons.
'''

def joints_dict():
    joints = {
        "coco": {
            "skeleton": [
                [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6]
            ]
        },
    }
    return joints


def draw_points(image, points, color=None):

    circle_size = max(2, int(np.sqrt(np.max(np.max(points, axis=0) - np.min(points, axis=0)) // 10)))
    for i, pt in enumerate(points):
        image = cv.circle(image, (int(pt[0]), int(pt[1])), circle_size, color[i], -1, lineType=cv.LINE_AA)

    return image


def draw_skeleton(image, points, skeleton, sk_color=None):

    canvas = copy.deepcopy(image)
    cur_canvas = canvas.copy()
    for i, joint in enumerate(skeleton):

        pt1, pt2 = points[joint]

        length = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(pt1[1] - pt2[1], pt1[0] - pt2[0]))
        polygon = cv.ellipse2Poly((int(np.mean((pt1[0], pt2[0]))), int(np.mean((pt1[1], pt2[1])))),
                                    (int(length / 2), 2), int(angle), 0, 360, 1)
        cv.fillConvexPoly(cur_canvas, polygon, sk_color[i], lineType=cv.LINE_AA)
        canvas = cv.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas


def draw_points_and_skeleton(image, points, skeleton):

    colors1 = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
               [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
               [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 85], [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
               [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
               [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 85]]
    image = draw_skeleton(image, points, skeleton, sk_color=colors1)
    image = draw_points(image, points, color=colors1)
    return image


def read_pickle(work_path):
    data_list = []
    with open(work_path, "rb") as f:
        while True:
            try:
                data = pickle.load(f)
                data_list.append(data)
            except EOFError:
                break
    return data_list


def image_out(pred, out_dir, gif_path, rescale = 1):
    skeleton = joints_dict()['coco']['skeleton']

    data = pred
    
    # padding
    kp = data * rescale  # [T,17,3]

    kp_x = kp[..., 0]
    kp_y = kp[..., 1]
    min_y = np.min(kp_y)
    min_x = np.min(kp_x)
    data[..., 0] = (kp_x - min_x)
    data[..., 1] = (kp_y - min_y)
    max_y = np.max(data[..., 0])
    max_x = np.max(data[..., 1])

    img = np.ones((int(max_x)+1, int(max_y)+1), dtype=np.uint8)  
    bgr_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    bgr_img[:, :, 0] = 255
    bgr_img[:, :, 1] = 255
    bgr_img[:, :, 2] = 255

    frames = []
    for i, pt in enumerate(data):
        frame = bgr_img.copy()
        frame = draw_points_and_skeleton(frame, pt, skeleton)
        frames.append(frame)
        cv.imwrite(out_dir + '/' + str(i) + '.png', frame)

    imageio.mimsave(gif_path, frames, duration=50)

def parse_option():

    #init
    parser = argparse.ArgumentParser(description="Visualization of input skeletons")
    #add
    parser.add_argument("--pkl_path",help="The dir of inpul [*.pkl] file.")
    parser.add_argument("--out_dir",help="The dir to store the images.")
    parser.add_argument("--out_gif_path",help="The gif file path, which is end with [.gif].")
    #groups
    opt = parser.parse_args()
    #return
    return opt


if __name__ == '__main__':
    
    opt = parse_option()
    data = read_pickle(opt.pkl_path)[0]
    image_out(data, opt.out_dir, opt.out_gif_path)
    print('ok')
