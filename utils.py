'''
This file contains functions to visualize the heatmap and detected bounding boxes
'''

import matplotlib
# matplotlib.use('tkagg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
import cv2
import torch

from scipy.ndimage.measurements import label


def save_boxes(args, recognized_boxes, recognized_scores, img_id):

    if len(recognized_scores) < 1 and len(recognized_boxes) < 1:
        return

    pdf_name = img_id.split("/")[0]
    math_csv_path = os.path.join(args.save_folder, args.exp_name, pdf_name + ".csv")

    if not os.path.exists(os.path.dirname(math_csv_path)):
        os.makedirs(os.path.dirname(math_csv_path))

    math_output = open(math_csv_path, 'a')

    recognized_boxes = np.concatenate((recognized_boxes,np.transpose([recognized_scores])),axis=1)

    page_num = int(img_id.split("/")[-1])

    col = np.array([int(page_num) - 1] * recognized_boxes.shape[0])
    math_regions = np.concatenate((col[:, np.newaxis], recognized_boxes), axis=1)

    np.savetxt(math_output, math_regions, fmt='%.2f', delimiter=',')
    math_output.close()

def draw_box (image, boxes):
    for b in boxes:
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)


def _img_to_tensor (image):
    rimg = cv2.resize(image, (512, 512), interpolation = cv2.INTER_AREA).astype(np.float32)
    # width = image.shape[0]
    # height = image.shape[1]
    # max_width = 1024
    # coef = max_width/width
    # new_width = int(width * coef)
    # new_height = int(height * coef)
    # rimg = cv2.resize(image, (new_height, new_width), interpolation = cv2.INTER_AREA).astype(np.float32)

    rimg -= np.array((246, 246, 246), dtype=np.float32)
    rimg = rimg[:, :, (2, 1, 0)]
    return torch.from_numpy(rimg).permute(2, 0, 1)


def FixImgCoordinates (images, boxes):
    new_boxes = []
    if isinstance(images, list):
            for i in range(len(images)):
                print(images[i].shape)
                bbs = []
                for o_box in boxes[i] :
                    b = [None] * 4
                    b[0] = int(o_box[0] * images[i].shape[0])
                    b[1] = int(o_box[1] * images[i].shape[1])
                    b[2] = int(o_box[2] * images[i].shape[0])
                    b[3] = int(o_box[3] * images[i].shape[1])
                    bbs.append(b)

                new_boxes.append(bbs)
    else:
        bbs = []
        for o_box in boxes[0] :
            b = [None] * 4
            b[0] = int(o_box[0] * images.shape[0]) 
            b[1] = int(o_box[1] * images.shape[1])
            b[2] = int(o_box[2] * images.shape[0])
            b[3] = int(o_box[3] * images.shape[1])
            bbs.append(b)

        new_boxes.append(bbs)

    return new_boxes


def DrawAllBoxes(images, boxes):

    for i in range(len(images)):
        draw_box(images[i], boxes[i])


def convert_to_binary(image):
    try:
        print(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(gray_image)
    except Exception as e:
        print(e)

    im_bw = np.zeros(gray_image.shape)
    im_bw[gray_image > 127] = 0
    im_bw[gray_image <= 127] = 1

    return im_bw



def read_math(args, pdf_name):

    math_file = os.path.join(args.math_dir, pdf_name + args.math_ext)
    data = np.array([])

    if os.path.exists(math_file):
        data = np.genfromtxt(math_file, delimiter=',')

        # if there is only one entry convert it to correct form required
        if len(data.shape) == 1:
            data = data.reshape(1, -1)

    return data


def voting_equal(votes, math_regions):
    # cast votes for the regions
    for box in math_regions:
        votes[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = \
            votes[int(box[1]):int(box[3]), int(box[0]):int(box[2])] + 1

    return votes


def convert_to_binary(image):
    try:
        print(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(gray_image)
    except Exception as e:
        print(e)

    im_bw = np.zeros(gray_image.shape)
    im_bw[gray_image > 127] = 0
    im_bw[gray_image <= 127] = 1

    return im_bw


def voting_algo(params):

    args, math_regions, pdf_name, page_num = params
    page_num = int(page_num)
    print('Processing ', pdf_name, ' > ', page_num)

    image = cv2.imread(os.path.join(args.home_images,pdf_name,str(int(page_num+1))+".jpg"))

    # vote for the regions
    original_width = image.shape[1]
    original_height = image.shape[0]
    thresh_votes = args.algo_threshold

    votes = np.zeros(shape=(original_height, original_width))
    votes = voting_equal(votes, math_regions)

    votes[votes < thresh_votes] = 0
    votes[votes >= thresh_votes] = 1

    # im_bw = convert_to_binary(image)
    structure = np.ones((3, 3), dtype=np.int)
    labeled, ncomponents = label(votes, structure)

    # found the boxes. Now extract the co-ordinates left,top,right,bottom
    boxes = []
    indices = np.indices(votes.shape).T[:, :, [1, 0]]

    for i in range(ncomponents):

        labels = (labeled == (i+1))
        pixels = indices[labels.T]

        if len(pixels) < 1:
            continue

        box = [min(pixels[:, 0]), min(pixels[:, 1]), max(pixels[:, 0]), max(pixels[:, 1])]

        # if args.postprocess:
        #     # expansion to correctly fit the region
        #     box = fit_box.adjust_box(im_bw, box)

        # if box has 0 width or height, do not add it in the final detections
        if feature_extractor.width(box) < 1 or feature_extractor.height(box) < 1:
            continue

        boxes.append(box)

    return boxes