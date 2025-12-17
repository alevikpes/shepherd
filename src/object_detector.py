import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

from shepherd.src.core.descriptors import Descriptor
from shepherd.src.core.detectors import Detector
from shepherd.src.core.matchers import Matcher
from shepherd.src.core.utils import (
    Colours16,
    image_file_to_ndarray,
)


class ObjectDetector:

    kp_ref = None
    des_ref = np.empty(0)
    kp_test = None
    des_test = np.empty(0)
    matches = None
    ref_img = None
    test_img = None
    test_video_path = ''
    test_video_codec = ''
    result_file_name = ''
    detector_type = ''
    descriptor_type = ''
    colour_scheme = 'gray'
    matcher_type = ''
    matcher = None
    extended = False
    bf_cross_check = False
    min_match_count = 20
    knn_distance = 0.7

    def __init__(
        self,
        *,
        project_path='',
        case_name='',
        case_type='image',
    ):
        """Defines an Object Detector class.

        Project directory must have subdirectories:
            - cases
            - input
            - output

        `case_path` has format `project_name/cases/case_name`.
        """
        self.load_data(project_path, case_name, case_type=case_type)

    def _load_case(self, project_path):
        # Load the case.
        spec = importlib.util.spec_from_file_location(
            'Config',
            f'{project_path}/config.py',
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules['Config'] = module
        spec.loader.exec_module(module)
        return module

    def load_data(self, project_path, case_name, case_type='image'):
        cfg = self._load_case(project_path)
        case = cfg.Case(case_name)
        INPUT_PATH = case.input_path
        REF_PATH = INPUT_PATH / 'ref'
        OUTPUT_PATH = case.output_path
        self.colour_scheme = case.colour_scheme
        print('Importing reference image ...')
        self.ref_img_path = case.ref_img
        if not Path(self.ref_img_path).exists():
            raise Exception(
                f'ERROR: File {str(self.ref_img_path)} does not exist!'
            )

        self.ref_img = image_file_to_ndarray(
            self.ref_img_path,
            self.colour_scheme,
        )
        DETECTION = case.detection
        self.detector_type = DETECTION.get('detector')
        self.descriptor_type = DETECTION.get('descriptor')
        MATCHER = DETECTION.get('matcher')
        self.matcher_type = MATCHER.get('type')
        self.extended = MATCHER.get('extended')
        self.k = MATCHER.get('k')
        self.bf_cross_check = MATCHER.get('bf_cross_check')
        self.knn_distance = MATCHER.get('knn_distance')
        self.min_match_count = MATCHER.get('min_match_count')
        self.matcher = Matcher(
            self.matcher_type,
            extended=self.extended,
            bf_cross_check=self.bf_cross_check,
        )
        print('Importing test image ...')
        TEST_PATH = INPUT_PATH / 'test'
        if case_type == 'image':
            TEST_IMAGE = case.test_image
            self.test_img_path = str(
                Path(TEST_PATH / TEST_IMAGE.get('img_file'))
            )
            if not Path(self.test_img_path).exists():
                raise Exception(
                    f'ERROR: File {str(self.test_img_path)} does not exist',
                )

            self.test_img = image_file_to_ndarray(
                self.test_img_path,
                self.colour_scheme,
            )
            self.result_file_name = OUTPUT_PATH / TEST_IMAGE.get('img_out_file')
        elif case_type == 'video':
            TEST_VIDEO = case.test_video
            if TEST_VIDEO.get('video_file') == 'webcam':
                self.test_video_path = 'webcam'
            else:
                self.test_video_path = str(
                    Path(INPUT_PATH / TEST_VIDEO.get('video_file'))
                )
                if not Path(self.test_video_path).exists():
                    raise Exception(
                        f'ERROR: File {str(self.test_video_path)} does not exist',
                    )

            #print(self.test_video_path)
            self.test_video_codec = case.test_video.get('codec')
            self.test_video_start_time = case.test_video.get('start_time')
            self.test_video_end_time = case.test_video.get('end_time')
            self.result_file_name = OUTPUT_PATH / TEST_VIDEO.get('video_out_file')

    def quick_match(
        self,
        descriptor_matcher_type=cv2.DescriptorMatcher_FLANNBASED,
    ):
        """

        :descriptor_matcher_type: Available types:
            FLANNBASED
            BRUTEFORCE
            BRUTEFORCE_L1
            BRUTEFORCE_HAMMING
            BRUTEFORCE_HAMMINGLUT
            BRUTEFORCE_SL2
        """
        #-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
        #minHessian = 400
        # Using SIFT, since SURF requires manual build of
        # opencv library (patented and non-free).
        detector = cv2.SIFT.create()
        self.kp_ref, self.des_ref = detector.detectAndCompute(self.ref_img, None)
        self.kp_test, self.des_test = detector.detectAndCompute(self.test_img, None)

        #-- Step 2: Matching descriptor vectors with a FLANN based matcher
        # Since SURF is a floating-point descriptor NORM_L2 is used
        matcher = cv2.DescriptorMatcher.create(descriptor_matcher_type)
        knn_matches = matcher.knnMatch(self.des_ref, self.des_test, 2)

        #-- Filter matches using the Lowe's ratio test
        self.matches = []
        for m, n in knn_matches:
            if m.distance < self.knn_distance * n.distance:
                self.matches.append(m)

        print(f'Found {len(self.matches)} matches.')

        ##-- Draw matches
        #img_matches = np.empty(
        #    (
        #        max(self.ref_img.shape[0], self.test_img.shape[0]),
        #        self.ref_img.shape[1]+self.test_img.shape[1],
        #        3
        #    ),
        #    dtype=np.uint8,
        #)
        #cv2.drawMatches(
        #    self.ref_img,
        #    self.kp_ref,
        #    self.test_img,
        #    self.kp_test,
        #    self.matches,
        #    img_matches,
        #    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        #)
        #return img_matches

        # Save points coordinates into separate arrays
        src_pts = np.zeros(self.min_match_count)
        dst_pts = np.zeros(self.min_match_count)
        #if len(self.matches) >= self.min_match_count:
        src_pts = np.float32(
            [self.kp_ref[m.queryIdx].pt for m in self.matches]
        )
        dst_pts = np.float32(
            [self.kp_test[m.trainIdx].pt for m in self.matches]
        )
        return src_pts, dst_pts

    def find_features(self, img: np.ndarray, verbose=False):
        """Find features in an image."""
        # Detect features
        detector = Detector(img, self.detector_type)
        kp = detector.detect()
        # Find descriptors
        descriptor = Descriptor(img, kp, self.descriptor_type)
        kp, des = descriptor.compute()

        if des is None:
            des = np.zeros((10, 10))

        if verbose:
            print(type(kp))
            print(type(des))
            print(f'kp: {kp}')
            print(f'des: {des}')

        return kp, des

    def get_good_matches(
        self,
        kp_ref=None,
        des_ref=np.empty(0),
        kp_test=None,
        des_test=np.empty(0),
        k=2,
        verbose=False,
    ):
        if (
            (
                isinstance(des_test, np.ndarray) and
                not all((kp_ref, des_ref.any(), kp_test, des_test.any()))
            ) or
            des_test is None
        ):
            self.kp_ref, self.des_ref = self.find_features(self.ref_img)
            self.kp_test, self.des_test = self.find_features(self.test_img)
        else:
            self.kp_ref = kp_ref
            self.des_ref = des_ref
            self.kp_test = kp_test
            self.des_test = des_test

        self.matches = self.matcher.match(
            self.des_ref,
            self.des_test,
            knn_distance=self.knn_distance,
            k=k,
            min_match_count=self.min_match_count,
            verbose=verbose,
        )

        # Save points coordinates into separate arrays
        src_pts = np.zeros(self.min_match_count)
        dst_pts = np.zeros(self.min_match_count)
        if len(self.matches) >= self.min_match_count:
            src_pts = np.float32(
                [self.kp_ref[m.queryIdx].pt for m in self.matches]
            )
            dst_pts = np.float32(
                [self.kp_test[m.trainIdx].pt for m in self.matches]
            )

        if verbose:
            Colours16.coloured_text(
                (
                    'Not enough matches found: '
                    f'{len(self.matches)}/{self.min_match_count}'
                ),
                Colours16.BOLD,
                Colours16.FG_RED,
                Colours16.BG_BLACK,
            )

        return src_pts, dst_pts

    def draw_features(self, img, pts, mark_features=False, limit=0):
        if not pts.all():
            return img

        if limit:
            pts = pts[:limit]

        for i in range(len(pts)):
            dp = pts[i]
            # draw circle of the dp coordinates
            # on the current frame
            circle_center = (int(dp[0]), int(dp[1]))  # coords of the centre
            cv2.circle(
                img,  # image
                circle_center,  # coords of the centre
                2,  # radius
                (0, 255, 0),  # colour green
                2,  # line thickness
            )
            if mark_features:
                text_start_x, text_start_y = int(dp[0]) - 7, int(dp[1]) - 7
                text = str(i)
                font = cv2.FONT_HERSHEY_TRIPLEX
                font_scale = 0.5
                font_thickness = 1
                # Draw text background.
                text_bg_colour = (255, 255, 255)
                text_size, _ = cv2.getTextSize(
                    text,
                    font,
                    font_scale,
                    font_thickness,
                )
                text_bg_top_left = (text_start_x - 1, text_start_y - 13)
                text_bg_bot_right = (
                    text_start_x + text_size[0] + 1,
                    text_start_y + text_size[1] - 9,
                )
                cv2.rectangle(
                    img,
                    text_bg_top_left,
                    text_bg_bot_right,
                    text_bg_colour,
                    -1,
                )
                # Draw text.
                params = (
                    img,
                    text,  # string to print
                    (text_start_x, text_start_y),  # coords of the left bottom / top corner of the text
                    font,
                    font_scale,  # font scale
                    (0, 0, 0),  # colour black
                    font_thickness,  # line thickness in pixels
                    cv2.LINE_AA,  # line type
                )
                cv2.putText(*params, bottomLeftOrigin=False)

        return img

    def draw_matches(self, matchesMask=None, only_keypoints=False, limit=0):
        #if only_keypoints:
        #    ref_img = self.ref_img
        #    test_img = self.test_img
        #    cv2.drawKeypoints(
        #        self.ref_img,
        #        self.kp_ref,
        #        ref_img,
        #        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        #    )
        #    cv2.drawKeypoints(
        #        self.test_img,
        #        self.kp_test,
        #        test_img,
        #        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        #    )
        #    # Concatenate images with equal dimensions.
        #    #img = cv2.hconcat((ref_img, test_img))
        #    #img = np.concatenate((ref_img, test_img), axis=1)

        #    # Concatenate images with different dimensions.
        #    h1, w1 = ref_img.shape[:2]
        #    h2, w2 = test_img.shape[:2]
        #    # Create an empty matrix for the resulting image
        #    img = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        #    # Place images into the new matrix
        #    img[:h1, :w1] = ref_img
        #    img[:h2, w1:w1 + w2] = test_img
        #    #return ref_img, test_img
        #else:
        img = cv2.drawMatches(
            self.ref_img,
            self.kp_ref,
            self.test_img,
            self.kp_test,
            self.matches,  #[:self.min_match_count],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        #elif self.matcher_type.lower() in ('brute_force_knn', 'flann'):
        #    #draw_params = dict(
        #    #    matchColor=(0, 255, 0),
        #    #    singlePointColor=(255, 0, 0),
        #    #    matchesMask=matchesMask,
        #    #    flags=cv.DrawMatchesFlags_DEFAULT,
        #    #)
        #    img = cv.drawMatchesKnn(
        #        self.ref_img,
        #        self.kp_ref,
        #        self.test_img,
        #        self.kp_test,
        #        self.matches,
        #        None,
        #        #**draw_params,
        #        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        #    )

        return img

    def draw_polygon(self, img, src_pts, dst_pts):
        """Draws a polygon around the detected object."""
        if not src_pts.all() or not dst_pts.all():
            return img

        M, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            # FIXME: research these methods.
            method=cv2.RANSAC,
            #method=cv2.LMEDS,
            #method=cv2.RHO,
            ransacReprojThreshold=3.0,  # Default is 3.0
            maxIters=2000,  # Default is 2000.
            confidence=0.995,  # Default is 0.995.
        )

        if self.colour_scheme == 'gray':
            # For gray images
            h, w = self.ref_img.shape
        elif self.colour_scheme == 'rgb':
            # For coloured images
            h, w, _ = img.shape

        # Get ref image corners.
        pts = np.float32(
            [[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]
        ).reshape(-1, 1, 2)
        if pts.any() and M is not None:
            # Draw a matching polygon
            dst = cv2.perspectiveTransform(pts, M)
            cv2.polylines(
                img,
                [np.int32(dst)],
                True,
                255,
                3,
                cv2.LINE_AA,
            )

        ## -- Get the corners from the ref image ( the object to be "detected" )
        #obj_corners = np.empty((4,1,2), dtype=np.float32)
        #obj_corners[0,0,0] = 0
        #obj_corners[0,0,1] = 0
        #obj_corners[1,0,0] = self.ref_img.shape[1]
        #obj_corners[1,0,1] = 0
        #obj_corners[2,0,0] = self.ref_img.shape[1]
        #obj_corners[2,0,1] = self.ref_img.shape[0]
        #obj_corners[3,0,0] = 0
        #obj_corners[3,0,1] = self.ref_img.shape[0]

        #if obj_corners.all() and M is not None:
        #    scene_corners = cv2.perspectiveTransform(obj_corners, M)

        #    #-- Draw lines between the corners (the mapped object in the scene - image_2 )
        #    cv2.line(
        #        img,
        #        (
        #            int(scene_corners[0,0,0]+self.ref_img.shape[1]),
        #            int(scene_corners[0,0,1]),
        #        ),
        #        (
        #            int(scene_corners[1,0,0]+self.ref_img.shape[1]),
        #            int(scene_corners[1,0,1]),
        #        ),
        #        (0, 255, 0),
        #        4,
        #    )
        #    cv2.line(
        #        img,
        #        (
        #            int(scene_corners[1,0,0]+self.ref_img.shape[1]),
        #            int(scene_corners[1,0,1]),
        #        ),
        #        (
        #            int(scene_corners[2,0,0]+self.ref_img.shape[1]),
        #            int(scene_corners[2,0,1]),
        #        ),
        #        (0, 255, 0),
        #        4,
        #    )
        #    cv2.line(
        #        img,
        #        (
        #            int(scene_corners[2,0,0]+self.ref_img.shape[1]),
        #            int(scene_corners[2,0,1]),
        #        ),
        #        (
        #            int(scene_corners[3,0,0]+self.ref_img.shape[1]),
        #            int(scene_corners[3,0,1]),
        #        ),
        #        (0, 255, 0),
        #        4,
        #    )
        #    cv2.line(
        #        img,
        #        (
        #            int(scene_corners[3,0,0]+self.ref_img.shape[1]),
        #            int(scene_corners[3,0,1]),
        #        ),
        #        (
        #            int(scene_corners[0,0,0]+self.ref_img.shape[1]),
        #            int(scene_corners[0,0,1]),
        #        ),
        #        (0, 255, 0),
        #        4,
        #    )

        return img

    def display_text(self, img, src_text):
        """Display text on the image.

        `src_text` must be in the format:
            scr_text = (
                f'Avg. FPS: {int(avg_fps)}',
                f'Detector: {DETECTOR_TYPE.upper()}',
                f'Descriptor: {DESCRIPTOR_TYPE.upper()}',
                f'KNN distance: {KNN_DISTANCE}',
                f'Features found: {len(good)}',
            )
        """
        text_params = [
            (
                img,  # image
                txt,  # string to print
                (5, (i + 1) * 15),  # coords of the left bottom / top corner of the text
                cv2.FONT_HERSHEY_DUPLEX,  # text font
                0.5,  # size multiplier
                (255, 255, 255),  # colour black
                1,  # line thickness in pixels
                #cv2.LINE_8,  # line type
                cv2.LINE_AA,  # line type
            ) for i, txt in enumerate(src_text)
        ]
        for params in text_params:
            cv2.putText(*params, bottomLeftOrigin=False)

        return img

    # FIXME: Finish implementation.
    def _set_text_params(self, text_params, bg_params={}):
        text_start_x, text_start_y = text_params.get('text_top_left')
        text = text_params.get('text_string')
        text_colour = text_params.get('colour')
        # Margins: (top, left, bottom, right)
        text_margins = text_params.get('margins')
        font = text_params.get('font')
        font_scale = text_params.get('font_scale')
        font_thickness = text_params.get('font_thickness')
        line_type = text_params.get('line_type')
        if bg_params:
            text_bg_colour = bg_params.get('colour')
            text_size, _ = cv2.getTextSize(
                text,
                font,
                font_scale,
                font_thickness,
            )
            text_bg_top_left = (
                text_start_x - text_margins[0],
                text_start_y - text_margins[1],
            )
            text_bg_bot_right = (
                text_start_x + text_size[0] + text_margins[2],
                text_start_y + text_size[1] + text_margins[3],
            )
            bg_params = (
                text_bg_top_left,
                text_bg_bot_right,
                text_bg_colour,
            )
        else:
            bg_params = tuple()

        text_params = (
            text,  # string to print
            (text_start_x, text_start_y),
            font,
            font_scale,  # font scale
            text_colour,
            font_thickness,  # line thickness in pixels
            line_type,
        )
        return text_params, bg_params
