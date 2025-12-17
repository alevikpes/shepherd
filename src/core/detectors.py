import cv2
import numpy as np


class Detector:

    def __init__(self, image: np.ndarray, detector_type):
        """
        Parameters:
        :image np.ndarray: - An image converted from file into a ndarray.
        :detector_type str: = Detector type can take one of:
            'orb',
            'brisk',
            'fast',
            'star'.
            'harris',
            'shi-tomasi',
            'sift',
            'surf',
            'kaze',
            'akaze', - fails atm.
        """
        self.image = image
        self.detector_type = detector_type.lower()

    def orb(self):
        """ Detector of the Oriented FAST and Rotated BRIEF (ORB)
        algorithm for features detection and description.

        ORB is a free alternative to SIFT and SURF and is basically a
        fusion of FAST keypoint detector and BRIEF descriptor with many
        modifications to enhance the performance.

        Parameters:
        :nfeatures int=500: - The maximum number of features to retain.
        :scaleFactor float=1.2: - Pyramid decimation ratio, greater than 1.
            scaleFactor == 2 means the classical pyramid, where each next
            level has 4x less pixels than the previous, but such a big scale
            factor will degrade feature matching scores dramatically. On
            the other hand, too close to 1 scale factor will mean that to
            cover certain scale range you will need more pyramid levels and
            so the speed will suffer.
        :nlevels int=8: - The number of pyramid levels. The smallest level
            will have linear size equal to
            input_image_linear_size/pow(scaleFactor, nlevels-firstLevel).
        :edgeThreshold int=31: - This is size of the border where the
            features are not detected. It should roughly match the
            patchSize parameter.
        :firstLevel int=0: - The level of pyramid to put source image to.
            Previous layers are filled with upscaled source image.
        :WTA_K int=2: - The number of points that produce each element of
            the oriented BRIEF descriptor. The default value 2 means the
            BRIEF where we take a random point pair and compare their
            brightnesses, so we get 0/1 response. Other possible values are
            3 and 4. For example, 3 means that we take 3 random points
            (of course, those point coordinates are random, but they are
            generated from the pre-defined seed, so each element of BRIEF
            descriptor is computed deterministically from the pixel
            rectangle), find point of maximum brightness and output index
            of the winner (0, 1 or 2). Such output will occupy 2 bits,
            and therefore it will need a special variant of Hamming
            distance, denoted as NORM_HAMMING2 (2 bits per bin). When
            WTA_K=4, we take 4 random points to compute each bin (that
            will also occupy 2 bits with possible values 0, 1, 2 or 3).
        :scoreType ORB.ScoreType=ORB.HARRIS_SCORE: - The default
            HARRIS_SCORE means that Harris algorithm is used to rank
            features (the score is written to KeyPoint::score and is used
            to retain best nfeatures features); FAST_SCORE is alternative
            value of the parameter that produces slightly less stable
            keypoints, but it is a little faster to compute.
        :patchSize int=31: - Size of the patch used by the oriented BRIEF
            descriptor. Of course, on smaller pyramid layers the perceived
            image area covered by a feature will be larger.
        :fastThreshold int=20: - The FAST threshold.
        """
        # Initiate ORB detector
        return cv2.ORB.create()

    def brisk(self):
        """ Binary Robust Invariant Scalable Keypoints (BRISK) - keypoint
        detector and descriptor extractor.

        Parameters:
        :thresh int=30: - AGAST detection threshold score.
        :octaves int=3: - Detection octaves. Use 0 to do single scale.
        :patternScale float=1: - Apply this scale to the pattern used
            for sampling the neighbourhood of a keypoint.
        :radiusList array: - Defines the radii (in pixels) where the
            samples around a keypoint are taken (for keypoint scale 1).
        :numberList array: - Defines the number of sampling points on the
            sampling circle. Must be the same size as radiusList..
        :dMax float=5.85: - Threshold for the short pairings used for
            descriptor formation (in pixels for keypoint scale 1).
        :dMin float=8.2: - Threshold for the long pairings used for
            orientation determination (in pixels for keypoint scale 1).
        :indexChange array: - Index remapping of the bits.
        """
        return cv2.BRISK.create()

    def star(self):
        """ STAR - experimental detector.

        Parameters:
        :maxSize int=45:
		:responseThreshold int=30:
		:lineThresholdProjected int=10:
        :lineThresholdBinarized int=8:
        :suppressNonmaxSize int=5:
        """
        return cv2.xfeatures2d.StarDetector.create()

    # https://docs.opencv.org/4.9.0/d5/d51/group__features2d__main.html#ga816d870cbdca71c6790c71bdf17df099
    def fast(self):
        """ `Features from Accelerated Segment Test` (FAST) algorithm for
        corner detection.

        Parameters:
        image numpy array: - Grayscale image where keypoints (corners)
            are detected.
        threshold int=10: - Threshold on difference between intensity
            of the central pixel and pixels of a circle around this pixel.
        nonmaxSuppression bool=True: - If true, non-maximum suppression
            is applied to detected corners (keypoints).
        type: - One of the three neighborhoods as defined in the paper:
            cv2.FAST_FEATURE_DETECTOR_TYPE_5_8,
            cv2.FAST_FEATURE_DETECTOR_TYPE_7_12,
            cv2.FAST_FEATURE_DETECTOR_TYPE_9_16 - default.
        """
        if len(self.image.shape) != 2:
            raise ValueError('Image type must be `gray`.')

        # Initiate FAST detector
        return cv2.FastFeatureDetector.create(nonmaxSuppression=False)

    # https://docs.opencv.org/4.9.0/dc/d0d/tutorial_py_features_harris.html
    def harris_corner(self, img, block_size, ksize, k):
        """ Harris Corner Detector.

        Parameters:
        :img numpy array: - Input image, it should be grayscale and float32
            type.
        :block_size int: - It is the size of neighbourhood considered for
            corner detection.
        :ksize int: - Aperture parameter of Sobel derivative used.
        :k float: - Harris detector free parameter in the equation.
        """
        if len(img.shape) != 2:
            raise ValueError('Image type must be `gray`')

        image = np.float32(img)
        dst = cv2.cornerHarris(image, block_size, ksize, k)
        dst = cv2.dilate(dst, None)  # Optional
        # Threshold for an optimal value, it may vary depending on the image. ???
        #img[dst>0.01*dst.max()] = [0, 0, 255]
        return dst

    # https://docs.opencv.org/4.9.0/d4/d8c/tutorial_py_shi_tomasi.html
    def shi_tomasi_corner(self, img, maxCorners, qualityLevel, minDistance):
        """ Shi-Tomasi Corner Detector.

        Parameters:
        :image numpy array: - Input 8-bit or floating-point 32-bit,
            single-channel image (gray).
        :corners int: - Output vector of detected corners.
        :maxCorners int: - Maximum number of corners to return. If there
            are more corners than are found, the strongest of them is
            returned. maxCorners <= 0 implies that no limit on the maximum
            is set and all detected corners are returned.
        :qualityLevel float: - Parameter characterizing the minimal accepted
            quality of image corners. The parameter value is multiplied
            by the best corner quality measure, which is the minimal
            eigenvalue (see cornerMinEigenVal) or the Harris function
            response (see cornerHarris). The corners with the quality
            measure less than the product are rejected. For example, if
            the best corner has the quality measure = 1500, and the
            qualityLevel=0.01 , then all the corners with the quality
            measure less than 15 are rejected.
        :minDistance float: - Minimum possible Euclidean distance between
            the returned corners.
        :mask numpy array: - Optional region of interest. If the image is
            not empty (it needs to have the type CV_8UC1 and the same size
            as image), it specifies the region in which the corners are
            detected. Default is None.
        :blockSize int: - Size of an average block for computing a derivative
            covariation matrix over each pixel neighborhood. See
            cornerEigenValsAndVecs. Default is 3.
        :useHarrisDetector bool: - Parameter indicating whether to use a
            Harris detector (see cornerHarris) or cornerMinEigenVal. Default
            is False.
        :k float: - Free parameter of the Harris detector. Default is 0.04.
        """
        if len(img.shape != 2):
            raise ValueError('Image type must be `gray`')

        stc = cv2.goodFeaturesToTrack(
            img, maxCorners, qualityLevel, minDistance)
        #return np.float32(stc)
        return np.int0(stc)

    # https://docs.opencv.org/4.9.0/da/df5/tutorial_py_sift_intro.html
    def sift(self):
        """ Extracting keypoints using the Scale Invariant Feature Transform.

        PATENTED!

        Parameters:
        :nfeatures int=0: - The number of best features to retain. The
            features are ranked by their scores (measured in SIFT algorithm
            as the local contrast).
        :nOctaveLayers int=3: - The number of layers in each octave. 3 is
            the value used in D. Lowe paper. The number of octaves is
            computed automatically from the image resolution.
        :contrastThreshold float=0.04: - The contrast threshold used to
            filter out weak features in semi-uniform (low-contrast)
            regions. The larger the threshold, the less features are
            produced by the detector. Note: The contrast threshold will
            be divided by nOctaveLayers when the filtering is applied.
            When nOctaveLayers is set to default and if you want to use
            the value used in D. Lowe paper, 0.03, set this argument to 0.09.
        :edgeThreshold float=10: - The threshold used to filter out
            edge-like features. Note that the its meaning is different
            from the contrastThreshold, i.e. the larger the edgeThreshold,
            the less features are filtered out (more features are retained).
        :sigma float=1.6: - The sigma of the Gaussian applied to the input
            image at the octave #0. If your image is captured with a weak
            camera with soft lenses, you might want to reduce the number.
        :enable_precise_upscale bool=False: - Whether to enable precise
            upscaling in the scale pyramid, which maps index x to 2x. This
            prevents localization bias. The option is disabled by default.
        """
        # Initiate SIFT detector
        return cv2.SIFT.create()

    # https://docs.opencv.org/4.10.0/df/dd2/tutorial_py_surf_intro.html#autotoc_md1271
    def surf(self, hessian_threshold=400, upright=False):
        """ Extracting `Speeded Up Robust Features` (SURF) from an image.

        PATENTED!

        Parameters:
        :hessianThreshold int=400: - Threshold for hessian keypoint detector
            used in SURF. Larger - less features detected. Recommended
            values in the range 300-500.
        :nOctaves int=4: - Number of pyramid octaves the keypoint detector
            will use. It is set to 4 by default. If you want to get very
            large features, use the larger value. If you want just small
            features, decrease it.
        :nOctaveLayers int=2: - Number of octave layers within each octave.
            It is set to 2 by default.
        :extended bool=False: - Extended descriptor flag (true - use extended
            128-element descriptors; false - use 64-element descriptors).
        :upright bool=False: - Up-right or rotated features flag (true - do not
            compute orientation of features, much faster computation;
            false - compute orientation).
        """
        # Initiate SURF detector
        return cv2.xfeatures2d.SURF.create(
            hessianThreshold=hessian_threshold,
            upright=upright,
        )

    def kaze(self):
        if len(self.image.shape) != 2:
            raise ValueError('Image type must be `gray`.')

        det = cv2.KAZE.create()
        return det

    def akaze(self):
        if len(self.image.shape) != 2:
            raise ValueError('Image type must be `gray`.')

        det = cv2.AKAZE.create()
        return det

    def detect(self):
        # Detect keypoints.
        # Optionally, `mask` can be used to specify the regions for detection.
        #print(f'Running {self.detector_type} detector ...')
        mask = None
        if self.detector_type == 'harris':
            return self.harris_corner(self.image, 2, 3, 0.04)
        elif self.detector_type == 'shi-tomasi':
            return self.shi_tomasi_corner(self.image, 25, 0.01, 10)
        elif self.detector_type in (
            'orb', 'brisk', 'sift', 'surf', 'fast', 'star', 'kaze', 'akaze'):
            if self.detector_type == 'orb':
                detector = self.orb()
            elif self.detector_type == 'brisk':
                detector = self.brisk()
            elif self.detector_type == 'sift':
                detector = self.sift()
            elif self.detector_type == 'surf':
                detector = self.surf()
            elif self.detector_type == 'fast':
                detector = self.fast()
            elif self.detector_type == 'star':
                detector = self.star()
            elif self.detector_type == 'kaze':
                detector = self.kaze()
            elif self.detector_type == 'akaze':
                detector = self.akaze()

            return detector.detect(self.image, mask)
        else:
            raise ValueError('Such detector type is not implemented.')
