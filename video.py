import cv2
import numpy as np
import dlib
import fonctions as f
from imutils import face_utils
import imutils

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# f.find_pt_bouche()
# f.find_pt_nez()
# f.find_pt_od()
# f.find_pt_og()

continu = True
while continu:
    _, image = cap.read()
    # image = imutils.resize(image, width=500)
    copy_img_origin = image.copy()
    copy_img_origin = copy_img_origin.astype(np.float32) / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 1)

    for (i, face) in enumerate(faces):
        forme = predictor(gray, face)
        forme = face_utils.shape_to_np(forme)

        pt_bouche_origine = f.get_pt_bouche_origne(forme)
        pt_nez_origine = f.get_pt_nez_origine(forme)
        pt_od_origine = f.get_pt_od_origine(forme)
        pt_og_origine = f.get_pt_og_origine(forme)

        copy_img_origin = f.add_bouche_face(pt_bouche_origine, copy_img_origin)
        copy_img_origin = f.add_nez_face(pt_nez_origine, copy_img_origin)
        copy_img_origin = f.add_od_face(pt_od_origine, copy_img_origin)
        copy_img_origin = f.add_og_face(pt_og_origine, copy_img_origin)

        cv2.imshow("image", copy_img_origin)

        if cv2.waitKey(33) == ord('a'):
            print("a pressed")    # a key to stop
            continu = False
            break
