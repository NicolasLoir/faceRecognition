from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import csv
import fonctions as f


# f.find_pt_bouche()
# f.find_pt_nez()
# f.find_pt_od()
# f.find_pt_og()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")


image = cv2.imread("./photo/exemple_old.jpg")
image = imutils.resize(image, width=500)
copy_img_origin = image.copy()
copy_img_origin = copy_img_origin.astype(np.float32) / 255.0
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_rec = detector(gray, 1)


for (i, rect) in enumerate(gray_rec):
    forme = predictor(gray, rect)
    forme = face_utils.shape_to_np(forme)

    image = f.add_rect_face(image, rect)
    image = f.add_circle_face(image, forme)
    f.showImg(image)


pt_bouche_origine = f.get_pt_bouche_origne(forme)
pt_nez_origine = f.get_pt_nez_origine(forme)
pt_od_origine = f.get_pt_od_origine(forme)
pt_og_origine = f.get_pt_og_origine(forme)


# img = f.add_circle(27, pt_nez_origine, image)
# f.showImg(img)


if (forme > 0).all():
    copy_img_origin = f.add_bouche_face(pt_bouche_origine, copy_img_origin)
    # f.showImg(copy_img_origin)

    # copy_img_origin = f.add_nez_face(pt_nez_origine, copy_img_origin)
    # f.showImg(copy_img_origin)

    # copy_img_origin = f.add_od_face(pt_od_origine, copy_img_origin)
    # f.showImg(copy_img_origin)

    # copy_img_origin = f.add_og_face(pt_og_origine, copy_img_origin)
    f.showImg(copy_img_origin)
