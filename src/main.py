import sys
from pathlib import Path

import cv2
import numpy as np

from shepherd.src.core.utils import (
    close_all_windows,
    get_case_name,
    img2file,
)
from shepherd.src.object_detector import ObjectDetector


def run_case(
    project_path,
    save_output=False,
    show_features=True,
    show_features_limit=0,
    mark_features=True,
    mark_object=True,
    show_matches=True,
    image_size=None,  #(3600, 1200),
):
    #PROJECT_PATH = DATA_PATH / 'saki_air_field'
    #CASE_NAME = 'case5_rgb'

    prj_path, case_name = get_case_name(project_path)

    od = ObjectDetector(
        project_path=prj_path,
        case_name=case_name,
    )
    #src_pts, dst_pts = od.get_good_matches()
    # Quick match.
    src_pts, dst_pts = od.quick_match()
    if not src_pts.any() or not dst_pts.any(): 
        print('No features found!')
        sys.exit(1)

    # Mark features on the images.
    if show_features:
        if mark_features:
            od.ref_img = od.draw_features(
                od.ref_img,
                src_pts,
                mark_features=True,
                limit=show_features_limit,
            )
            od.test_img = od.draw_features(
                od.test_img,
                dst_pts,
                mark_features=True,
                limit=show_features_limit,
            )
        else:
            od.ref_img = od.draw_features(
                od.ref_img,
                src_pts,
                limit=show_features_limit,
            )
            od.test_img = od.draw_features(
                od.test_img,
                dst_pts,
                limit=show_features_limit,
            )

    # Add text to the image
    od.test_img = od.display_text(
        od.test_img,
        (
            f'Detector: {od.detector_type.upper()}',
            f'Descriptor: {od.descriptor_type.upper()}',
            f'Matcher: {od.matcher_type.upper()}',
        ),
    )

    # Draw frame on the test image.
    if mark_object:
        od.test_img = od.draw_polygon(od.test_img, src_pts, dst_pts)

    # Draw matches
    if show_matches:
        img = od.draw_matches(limit=show_features_limit)
    else:
        #img = od.draw_matches(only_keypoints=True)
        # Concatenate images with different dimensions.
        h1, w1 = od.ref_img.shape[:2]
        h2, w2 = od.test_img.shape[:2]
        # Create an empty matrix for the resulting image
        img = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        # Place images into the new matrix
        img[:h1, :w1] = od.ref_img
        img[:h2, w1:w1 + w2] = od.test_img

    #cv2.imshow('Temp', img)
    # Save image to file.
    if save_output:
        img2file(img, od.result_file_name, verbose=True)

    # Set image size.
    if image_size:
        img = cv2.resize(img, image_size)

    return img


if __name__ == '__main__':
    DATA_PATH = Path(__file__).parent.parent / 'data'
    PROJECT_PATH = DATA_PATH / 'bike/cases/case1_rgb.json'

    img = run_case(
        PROJECT_PATH,
        save_output=True,  # Save output image to a file.
        show_features=True,  # Show features on the image.
        show_features_limit=50,  # Show this number of features.
        mark_features=False,  # Show features ids on the image.
        mark_object=True,  # Show rectangle around the deected object.
        show_matches=False,  # Show matching lines on the image.
        image_size=(2400, 1000),
    )

    # Show image with OpenCV.
    win_name = 'Matches'
    cv2.imshow(win_name, img)
    close_all_windows(win_name)
