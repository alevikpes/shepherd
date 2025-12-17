import cv2
import numpy as np

from shepherd.src.core.detectors import Detector


class Descriptor:

    def __init__(self, image: np.ndarray, keypoints, descriptor_type):
        """
        Parameters:
        :image np.ndarray: - Must be the same image from which the keypoints
            were detected.
        :keypoints: = Detected features.
        :descriptor_type str: - The algorithm used for features descriptions.
            Possible values are
                'brief',
                'brisk',
                'orb',
                'sift',
                'surf',
                'akaze', - fails atm.
        """
        self.image = image
        self.keypoints = keypoints
        self.descriptor_type = descriptor_type.lower()

    # https://docs.opencv.org/4.9.0/dc/d7d/tutorial_py_brief.html
    def brief(self):
        """ Binary Robust Independent Elementary Features (BRIEF).

        BRIEF is a feature descriptor, it doesn't provide any method
        to find the features. So you will have to use any other feature
        detectors. Works best with STAR.

        Parameters:
        :bytes int=32: - Length of the descriptor in bytes, valid values
            are: 16, 32 (default) or 64.
        :use_orientation bool=False: - Sample patterns using keypoints
            orientation, disabled by default.
        """
        return cv2.xfeatures2d.BriefDescriptorExtractor.create(
            use_orientation=True,
        )

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

    def orb(self):
        """ Descriptor of the Oriented FAST and Rotated BRIEF (ORB)
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
        # Initiate ORB descriptor
        return cv2.ORB.create()

    # https://docs.opencv.org/4.9.0/da/df5/tutorial_py_sift_intro.html
    def sift(self):
        """ Computing descriptors for previously detected keypoints using the
        Scale Invariant Feature Transform.

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
    def surf(self):
        """ Computing descriptors for the previosly extracted keypoints using
        `Speeded Up Robust Features` (SURF).

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
        return cv2.xfeatures2d.SURF_create()

    def kaze(self):
        return cv2.KAZE.create()

    def akaze(self):
        return cv2.AKAZE.create()

    def compute(self):
        """ Compute descriptors for the given keypoints.

        Returns a tuple (keypoints, descriptors).
        """
        #print(f'Running {self.descriptor_type} descriptor ...')
        if self.descriptor_type == 'brief':
            descriptor = self.brief()
        elif self.descriptor_type == 'brisk':
            descriptor = self.brisk()
        elif self.descriptor_type == 'orb':
            descriptor = self.orb()
        elif self.descriptor_type == 'sift':
            descriptor = self.sift()
        elif self.descriptor_type == 'surf':
            descriptor = self.surf()
        elif self.descriptor_type == 'kaze':
            descriptor = self.kaze()
        elif self.descriptor_type == 'akaze':
            descriptor = self.akaze()
        else:
            raise ValueError(
                'Such descriptor type is not implemented.')

        return descriptor.compute(self.image, self.keypoints)

    # NOTE: testing
    #def quick_compute(self, detector_type):
    #    """Quick compute descriptors for the given keypoints.

    #    NOTE: Descriptors and detectors call the same `create` method.
    #    So, what's the point doing it twice?

    #    Returns a tuple (keypoints, descriptors).
    #    """
    #    #print(f'Running {self.descriptor_type} descriptor ...')
    #    det = Detector(self.image, detector_type)
    #    if self.descriptor_type == 'brief':
    #        detector = det.brief()
    #    elif self.descriptor_type == 'brisk':
    #        detector = det.brisk()
    #    elif self.descriptor_type == 'orb':
    #        detector = det.orb()
    #    elif self.descriptor_type == 'star':
    #        detector = det.star()
    #    elif self.descriptor_type == 'fast':
    #        detector = det.fast()
    #    elif self.descriptor_type == 'sift':
    #        detector = det.sift()
    #    elif self.descriptor_type == 'surf':
    #        detector = det.surf()
    #    elif self.descriptor_type == 'kaze':
    #        detector = det.kaze()
    #    elif self.descriptor_type == 'akaze':
    #        detector = det.akaze()
    #    else:
    #        raise ValueError(
    #            'Such detector type is not implemented.')

    #    return detector.detectAndCompute(self.image, None)
