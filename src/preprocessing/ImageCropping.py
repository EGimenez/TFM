# python ImageCropping.py
# This code is an addaptation of align_face code of Glow: Generative Flow with Invertible 1x1 Convolutions

# import the necessary packages
from PIL import Image
import numpy as np
import imutils
import cv2
from time import time
import argparse
from utils.test_dir import save_make_dir

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('--index', default='../../data/celeba/index/list_landmarks_celeba.txt', help='path to the list_landmarks_celeba.txt')
ap.add_argument("--input_path", default='../../data/celeba_wild/data_raw/', help='path to the wild images')
ap.add_argument("--output_path", default='../../data/celeba_wild/data_256_fast/', help='path to the 256x256 images ')
args = ap.parse_args()


class MyFaceAligner:
    def __init__(self, desiredLeftEye=(0.35, 0.35), desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, leftEyeCenter, rightEyeCenter):
        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                      (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output


my_fa = MyFaceAligner(desiredFaceWidth=256,
                      desiredLeftEye=(0.371, 0.480))


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

            rightEye = (int(line[1]), int(line[2]))
            leftEye = (int(line[3]), int(line[4]))

            try:
                img = Image.open(input_path + file)
                img = img.convert('RGB')  # if image is RGBA or Grayscale etc

                h = img.height
                w = img.width

                img = np.array(img)
                img = imutils.resize(img, width=800)

                h_ratio = img.shape[0] / h
                w_ratio = img.shape[1] / w

                leftEye = ((h_ratio * leftEye[0]) // 1, (w_ratio * leftEye[1]) // 1)
                rightEye = ((h_ratio * rightEye[0]) // 1, (w_ratio * rightEye[1]) // 1)

                align_img = my_fa.align(img, leftEye, rightEye )
                img = Image.fromarray(align_img, 'RGB')
                img.save(output_path + file)
            except:
                print(file)

    t_1 = time()
    print(t_1 -t_0)
