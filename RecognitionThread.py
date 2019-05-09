#!/usr/bin/env python

import threading
import time
import numpy as np
import os
from collections import namedtuple
import cv2
import dlib
import keras
from keras.utils.generic_utils import CustomObjectScope

class RecognitionThread(threading.Thread):

    def __init__(self, parent, params):
        print("Initializing recognition thread...")
        threading.Thread.__init__(self)
        self.parent = parent

        ##### Initialize aligners for face alignment.
        aligner_path = params.get("recognition", "aligner")
        aligner_targets_path = params.get("recognition", "aligner_targets")
        self.aligner = dlib.shape_predictor(aligner_path)

        # load targets
        aligner_targets = np.loadtxt(aligner_targets_path)
        left_eye = (aligner_targets[36] + aligner_targets[39]) / 2
        right_eye = (aligner_targets[42] + aligner_targets[45]) / 2
        nose = aligner_targets[30]
        left_mouth = aligner_targets[48]
        right_mouth = aligner_targets[54]
        self.shape_targets = np.stack((left_eye, left_mouth, nose, right_eye, right_mouth))

        ##### Initialize networks for Age, Gender and Expression
        ##### 1. AGE, GENDER, SMILE MULTITASK
        print("Initializing multitask network...")
        multitaskpath = params.get("recognition", "multitask_folder")
        with CustomObjectScope({'relu6': keras.layers.ReLU(6.),
                                'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
            self.multiTaskNet = keras.models.load_model(os.path.join(multitaskpath, 'model.h5'))
        self.multiTaskNet._make_predict_function()

        ##### Read class names
        self.expressions = {int(key): val for key, val in params['expressions'].items()}  # convert string key to int
        self.minDetections = int(params.get("recognition", "mindetections"))

        # Starting the thread
        self.switching_model = False
        self.recognition_running = False
        print("Recognition thread started...")

    def crop_face(self, img, rect, margin=0.2):
        x1 = rect.left()
        x2 = rect.right()
        y1 = rect.top()
        y2 = rect.bottom()
        # size of face
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        # Extend the area into square shape:
        if w > h:
            center = int(0.5 * (y1 + y2))
            h = w
            y1 = center - int(h / 2)
            y2 = y1 + h
        elif h > w:
            center = int(0.5 * (x1 + x2))
            w = h
            x1 = center - int(w / 2)
            x2 = x1 + w

        # add margin
        full_crop_x1 = x1 - int(w * margin)
        full_crop_y1 = y1 - int(h * margin)
        full_crop_x2 = x2 + int(w * margin)
        full_crop_y2 = y2 + int(h * margin)
        # size of face with margin
        new_size_w = full_crop_x2 - full_crop_x1 + 1
        new_size_h = full_crop_y2 - full_crop_y1 + 1

        # ensure that the region cropped from the original image with margin
        # doesn't go beyond the image size
        crop_x1 = max(full_crop_x1, 0)
        crop_y1 = max(full_crop_y1, 0)
        crop_x2 = min(full_crop_x2, img.shape[1] - 1)
        crop_y2 = min(full_crop_y2, img.shape[0] - 1)
        # size of the actual region being cropped from the original image
        crop_size_w = crop_x2 - crop_x1 + 1
        crop_size_h = crop_y2 - crop_y1 + 1

        # coordinates of region taken out of the original image in the new image
        new_location_x1 = crop_x1 - full_crop_x1
        new_location_y1 = crop_y1 - full_crop_y1
        new_location_x2 = crop_x1 - full_crop_x1 + crop_size_w - 1
        new_location_y2 = crop_y1 - full_crop_y1 + crop_size_h - 1

        new_img = np.random.randint(256, size=(new_size_h, new_size_w, img.shape[2])).astype('uint8')

        new_img[new_location_y1: new_location_y2 + 1, new_location_x1: new_location_x2 + 1, :] = \
            img[crop_y1:crop_y2 + 1, crop_x1:crop_x2 + 1, :]

        # if margin goes beyond the size of the image, repeat last row of pixels
        if new_location_y1 > 0:
            new_img[0:new_location_y1, :, :] = np.tile(new_img[new_location_y1, :, :], (new_location_y1, 1, 1))

        if new_location_y2 < new_size_h - 1:
            new_img[new_location_y2 + 1:new_size_h, :, :] = np.tile(new_img[new_location_y2:new_location_y2 + 1, :, :],
                                                                    (new_size_h - new_location_y2 - 1, 1, 1))
        if new_location_x1 > 0:
            new_img[:, 0:new_location_x1, :] = np.tile(new_img[:, new_location_x1:new_location_x1 + 1, :],
                                                       (1, new_location_x1, 1))
        if new_location_x2 < new_size_w - 1:
            new_img[:, new_location_x2 + 1:new_size_w, :] = np.tile(new_img[:, new_location_x2:new_location_x2 + 1, :],
                                                                    (1, new_size_w - new_location_x2 - 1, 1))

        return new_img

    def five_points_aligner(self, rect, shape_targets, landmarks_pred, img):

        B = shape_targets
        A = np.hstack((np.array(landmarks_pred), np.ones((len(landmarks_pred), 1))))

        a = np.row_stack((np.array([-A[0][1], -A[0][0], 0, -1]), np.array([
            A[0][0], -A[0][1], 1, 0])))
        b = np.row_stack((-B[0][1], B[0][0]))

        for i in range(A.shape[0] - 1):
            i += 1
            a = np.row_stack((a, np.array([-A[i][1], -A[i][0], 0, -1])))
            a = np.row_stack((a, np.array([A[i][0], -A[i][1], 1, 0])))
            b = np.row_stack((b, np.array([[-B[i][1]], [B[i][0]]])))

        X, res, rank, s = np.linalg.lstsq(a, b, rcond=-1)
        cos = (X[0][0]).real.astype(np.float32)
        sin = (X[1][0]).real.astype(np.float32)
        t_x = (X[2][0]).real.astype(np.float32)
        t_y = (X[3][0]).real.astype(np.float32)

        H = np.array([[cos, -sin, t_x], [sin, cos, t_y]])
        s = np.linalg.eigvals(H[:, :-1])
        R = s.max() / s.min()

        if R < 2.0:
            warped = cv2.warpAffine(img, H, (224, 224))
        else:
            # Seems to distort too much, probably error in landmarks
            # Let's just crop.
            crop = self.crop_face(img, rect)
            warped = cv2.resize(crop, (224, 224))

        return warped

    def run(self):
        Celebinfo = namedtuple('Celeb', ['filename', 'distance'])

        while not self.parent.isTerminated():

            while self.switching_model:
                self.recognition_running = False
                time.sleep(0.1)

            self.recognition_running = True

            faces = self.parent.getFaces()
            while faces == None:
                time.sleep(0.1)
                faces = self.parent.getFaces()

            validFaces = [f for f in faces if len(f['bboxes']) > self.minDetections]

            for face in validFaces:
                # get the timestamp of the most recent frame:
                timestamp = face['timestamps'][-1]
                unit = self.parent.getUnit(self, timestamp)

                if unit is not None:
                    img = unit.getFrame()
                    mean_box = np.mean(face['bboxes'], axis=0)
                    x, y, w, h = [int(c) for c in mean_box]

                    # Align the face to match the targets

                    # 1. DETECT LANDMARKS
                    dlib_box = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)
                    dlib_img = img[..., ::-1].astype(np.uint8)  # BGR to RGB
                    s = self.aligner(dlib_img, dlib_box)
                    landmarks = [[s.part(k).x, s.part(k).y] for k in range(s.num_parts)]

                    # 2. ALIGN
                    landmarks = np.array(landmarks)
                    crop = self.five_points_aligner(dlib_box, self.shape_targets, landmarks,
                                                    dlib_img)

                    # Save aligned face crop, used for debugging if turned on.
                    face["crop"] = crop

                    crop = crop.astype(np.float32)

                    # Preprocess network inputs, add singleton batch dimension
                    recog_input = np.expand_dims(crop / 255, axis=0)

                    # Recognize age, gender and smile in one forward pass

                    ageout, genderout, smileout = self.multiTaskNet.predict(recog_input)
                    age = np.dot(ageout[0], list(range(101)))
                    if "age" in face:
                        face["age"] = 0.95 * face["age"] + 0.05 * age
                    else:
                        face["age"] = age
                        face["recog_round"] = 0

                    gender = genderout[0][1]  # male probability
                    if "gender" in face:
                        face["gender"] = 0.8 * face["gender"] + 0.2 * gender
                    else:
                        face["gender"] = gender

                    t = smileout[0]
                    t = np.argmax(t)
                    expression = self.expressions[t]
                    face["expression"] = expression

                    face["recog_round"] += 1