# python ImageCropping.py
# This code is an addaptation of align_face code of Glow: Generative Flow with Invertible 1x1 Convolutions

# import the necessary packages
from imutils.face_utils import FaceAligner
from PIL import Image
import numpy as np
import imutils
import dlib
import cv2
from time import time
import argparse
from utils.test_dir import save_make_dir

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('--index', default='../../data/celeba/index/list_landmarks_celeba.txt', help='path to the list_landmarks_celeba.txt')
ap.add_argument("--input_path", default='../../data/celeba_wild/data_raw/', help='path to the wild images')
ap.add_argument("--output_path", default='../../data/celeba_wild/data_256/', help='path to the 256x256 images ')
ap.add_argument("--shape_predictor", default='shape_predictor_68_face_landmarks.dat', help="path to facial landmark predictor")
args = ap.parse_args()


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args.shape_predictor)
fa = FaceAligner(predictor, desiredFaceWidth=256,
                 desiredLeftEye=(0.371, 0.480))


# Input: numpy array for image with RGB channels
# Output: (numpy array, face_found)
def align_face(img):
    img = img[:, :, ::-1]  # Convert from RGB to BGR format
    img = imutils.resize(img, width=800)

    # detect faces in the grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)

    if len(rects) > 0:
        # align the face using facial landmarks
        align_img = fa.align(img, gray, rects[0])[:, :, ::-1]
        align_img = np.array(Image.fromarray(align_img).convert('RGB'))
        return align_img, True
    else:
        # No face found
        return None, False


# Input: img_path
# Output: aligned_img if face_found, else None
def align(img_path):
    img = Image.open(img_path)
    img = img.convert('RGB')  # if image is RGBA or Grayscale etc
    img = np.array(img)
    x, face_found = align_face(img)
    return x


if __name__ == '__main__':
    index_path = args.index
    input_path = args.input_path
    output_path = args.output_path

    save_make_dir(input_path)
    save_make_dir(output_path)

    with open(index_path) as fp:

        # The first two are use less
        line = fp.readline()
        line = fp.readline()

        t_0 = time()

        while line:
            line = fp.readline()
            line = line.split()
            file = line[0]

            x = align(input_path + file)
            try:
                 img = Image.fromarray(x, 'RGB')
                 img.save(output_path + file)
            except:
                 print(file)
    t_1 = time()
    print(t_1 -t_0)
