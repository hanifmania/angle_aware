#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import os
from PIL import Image
import img2pdf


def generate_ar_markers():
    dpi = 72
    marker_mm = 150
    ar_type = (
        cv2.aruco.DICT_4X4_250
    )  # AR markerの種類(https://docs.opencv.org/3.4/d9/d6a/group__aruco.html#gac84398a9ed9dd01306592dd616c2c975)
    id_start = 0
    id_end = 50 + 1
    side_buffer_mm = 40
    str_start_mm = 5
    str_font_scale = 0.5
    str_thickness = 1
    pdf_name = "ar_map.pdf"
    output_file = os.path.dirname(__file__) + "/../data/ar_markers/"

    inch_per_mm = 1.0 / 25.4 * 1.05  ## [WARN] なぜが1.05倍すると上手く行く

    dictionary = cv2.aruco.getPredefinedDictionary(ar_type)
    pixel_per_mm = dpi * inch_per_mm
    print("error_max_mm: {}".format(1.0 / pixel_per_mm))
    ar_size_pixel = marker_mm * pixel_per_mm
    side_buffer_pixel = side_buffer_mm * pixel_per_mm
    str_buffer_pixel = str_start_mm * pixel_per_mm
    str_start_pixel = str_buffer_pixel + ar_size_pixel + side_buffer_pixel
    ar_size_pixel = int(ar_size_pixel)
    side_buffer_pixel = int(side_buffer_pixel)
    str_start_pixel = int(str_start_pixel)
    str_buffer_pixel = int(str_buffer_pixel)

    white = [255, 255, 255]
    black = [0, 0, 255]

    img_paths = []
    for id in range(id_start, id_end):

        ar_marker = cv2.aruco.drawMarker(dictionary, id, int(ar_size_pixel))
        print(ar_marker.shape)
        # rgb_img = cv2.cvtColor(ar_marker, cv2.COLOR_GRAY2RGB)
        img_pad = cv2.copyMakeBorder(
            ar_marker,
            0,
            0,
            int(side_buffer_pixel),
            int(side_buffer_pixel),
            cv2.BORDER_CONSTANT,
            value=white,
        )
        id_str = "ID : {:04d}".format(id)
        img = cv2.putText(
            img_pad,
            id_str,
            (str_start_pixel, ar_size_pixel - str_buffer_pixel),
            cv2.FONT_HERSHEY_SIMPLEX,
            str_font_scale,
            black,
            str_thickness,
            cv2.LINE_AA,
        )
        output_path = output_file + "/{:04d}.jpg".format(id)
        PILimage = Image.fromarray(img)
        ### add dpi information
        PILimage.save(output_path, dpi=(dpi, dpi))
        img_paths.append(output_path)

    pdf_bytes = img2pdf.convert(img_paths)
    pdf_path = output_file + pdf_name
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)


if __name__ == "__main__":
    generate_ar_markers()
