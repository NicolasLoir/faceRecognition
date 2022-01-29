import cv2
from imutils import face_utils
import numpy as np
import csv

# copy_img_origin = copy_img_origin * 255
# to save: cv2.imwrite("copy_img_origin_final.jpg", image)s


def showImg(image):
    cv2.imshow('Image.jpg', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def add_rect_face(image, rect):
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    # print((x, y, w, h))
    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image


def add_circle(begin_index, list_point, img):
    # print(begin_index)
    # print(list_point)
    landmarks = [None] * 100
    for (x, y) in list_point:
        # print(j)
        # print(landmarks[j])
        center_coordinates = (x, y)
        radius = 3
        color = (255, 0, 0)
        thickness = 1
        image = cv2.circle(img, center_coordinates, radius, color, thickness)
        # image = cv2.circle(img, (x, y), 3, (255, 0, 0), 1)

        image = cv2.putText(image, str(begin_index), (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        begin_index = begin_index+1
        # cv2.imshow("img", image)
        # cv2.waitKey(0)
    return image


def add_circle_face(image, forme):
    begin_index = 0
    image = add_circle(begin_index, forme, image)
    return image


def get_pt_bouche_origne(forme):
    return np.array(
        [
            forme[48],
            forme[49],
            forme[50],
            forme[51],
            forme[52],
            forme[53],
            forme[54],
            forme[55],
            forme[56],
            forme[57],
            forme[58],
            forme[59],
        ]
    )


def get_pt_nez_origine(forme):
    return np.array(
        [
            forme[27],
            forme[28],
            forme[29],
            forme[30],
            forme[31],
            forme[32],
            forme[33],
            forme[34],
            forme[35],
        ],
        dtype="float32",
    )


def get_pt_od_origine(forme):
    return np.array(
        [
            forme[42],
            forme[43],
            forme[44],
            forme[45],
            forme[46],
            forme[47],
        ],
        dtype="float32",
    )


def get_pt_og_origine(forme):
    return np.array(
        [
            forme[36],
            forme[37],
            forme[38],
            forme[39],
            forme[40],
            forme[41],
        ],
        dtype="float32",
    )


def convert_csv_mask_to_array(path_mask_csv):
    with open(path_mask_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        src_pts = []
        for i, row in enumerate(csv_reader):
            # Si presence de ligne vide alors on passe a la suite
            try:
                src_pts.append(np.array([int(row[1]), int(row[2])]))
            except ValueError:
                continue
    src_pts = np.array(src_pts, dtype="int")
    return src_pts


def add_element_face(path_img_ajouter, pt_elmt_ajouter, pt_elmt_origine, copy_img_origin):
    print("path_img_ajouter")
    print(path_img_ajouter)
    print("pt_elmt_ajouter")
    print(pt_elmt_ajouter)
    print("pt_elmt_origine")
    print(pt_elmt_origine)
    print("copy_img_origin")
    print(copy_img_origin)
    image_ajouter = cv2.imread(path_img_ajouter, cv2.IMREAD_UNCHANGED)
    image_ajouter = image_ajouter.astype(np.float32)
    image_ajouter = image_ajouter / 255.0

    # showImg(image_ajouter)

    M, _ = cv2.findHomography(pt_elmt_ajouter, pt_elmt_origine)

    max_width = copy_img_origin.shape[1]
    max_height = copy_img_origin.shape[0]

    transformed_mask = cv2.warpPerspective(
        image_ajouter,
        M,
        (max_width, max_height),
        None,
        cv2.INTER_LINEAR,
        cv2.BORDER_CONSTANT,
    )

    # showImg(transformed_mask)

    alpha_mask = transformed_mask[:, :, 3]
    # showImg(alpha_mask)
    alpha_image = 1 - alpha_mask
    # showImg(alpha_image)

    for c in range(0, 3):
        copy_img_origin[:, :, c] = (
            alpha_mask * transformed_mask[:, :, c]
            + alpha_image * copy_img_origin[:, :, c]
        )
        # showImg(copy_img_origin)

    # copy_img_origin = copy_img_origin * 255
    # cv2.imwrite("copy_img_origin_final.jpg", copy_img_origin)

    # showImg(copy_img_origin)
    return copy_img_origin


def add_bouche_face(pt_bouche_origine, copy_img_origin):
    path_bouche_csv = "./masks/bouche.csv"
    pt_bouche_ajouter = convert_csv_mask_to_array(path_bouche_csv)

    path_bouche_ajouter = "./masks/bouche.png"
    img = add_element_face(path_bouche_ajouter,
                           pt_bouche_ajouter, pt_bouche_origine, copy_img_origin)
    return img


def add_od_face(pt_od_origine, copy_img_origin):
    path_od_csv = "./masks/od.csv"
    pt_od_ajouter = convert_csv_mask_to_array(path_od_csv)

    path_od_ajouter = "./masks/od.png"
    img = add_element_face(path_od_ajouter,
                           pt_od_ajouter, pt_od_origine, copy_img_origin)
    return img


def add_og_face(pt_od_origine, copy_img_origin):
    path_og_csv = "./masks/og.csv"
    pt_og_ajouter = convert_csv_mask_to_array(path_og_csv)

    path_og_ajouter = "./masks/og.png"
    img = add_element_face(path_og_ajouter,
                           pt_og_ajouter, pt_od_origine, copy_img_origin)
    return img


def add_nez_face(pt_nez_origine, copy_img_origin):
    path_nez_csv = "./masks/nez.csv"
    pt_nez_ajouter = convert_csv_mask_to_array(path_nez_csv)

    path_nez_ajouter = "./masks/nez.png"
    img = add_element_face(path_nez_ajouter,
                           pt_nez_ajouter, pt_nez_origine, copy_img_origin)
    return img


def find_pt(begin_index, path_csv_ajouter, path_img_ajouter):
    pt_elmt_ajouter = convert_csv_mask_to_array(path_csv_ajouter)
    image_elmt_ajouter = cv2.imread(path_img_ajouter)

    image_elmt_ajouter = add_circle(
        begin_index, pt_elmt_ajouter, image_elmt_ajouter)
    showImg(image_elmt_ajouter)


def find_pt_bouche():
    path_bouche_csv = "./masks/bouche.csv"
    path_bouche_ajouter = "./masks/bouche.png"
    begin_index = 49
    find_pt(begin_index, path_bouche_csv, path_bouche_ajouter)


def find_pt_nez():
    path_nez_csv = "./masks/nez.csv"
    path_nez_ajouter = "./masks/nez.png"
    begin_index = 28
    find_pt(begin_index, path_nez_csv, path_nez_ajouter)


def find_pt_od():
    path_od_csv = "./masks/od.csv"
    path_od_ajouter = "./masks/od.png"
    begin_index = 43
    find_pt(begin_index, path_od_csv, path_od_ajouter)


def find_pt_og():
    path_og_csv = "./masks/og.csv"
    path_og_ajouter = "./masks/og.png"
    begin_index = 37
    find_pt(begin_index, path_og_csv, path_og_ajouter)
