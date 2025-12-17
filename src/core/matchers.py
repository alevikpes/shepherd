from operator import attrgetter

import cv2


FLANN_INDEX_KDTREE = 1  # For SIFT, SURF etc.
FLANN_INDEX_LSH = 6  # For ORB.
KNN_DISTANCE = 0.7  # Distance for the KNN ratio test. Determined from the case config.
BF_BEST_MATCHES = 20  # First best matches from the standard Brute-Force match.


class Matcher:

    _BF_NORM_TYPES = {
        'bf_l2': cv2.NORM_L2,  # For SIFT and SURF descriptors
        'bf_l1': cv2.NORM_L1,  # For SIFT and SURF descriptors
        'bf_h': cv2.NORM_HAMMING,  # For ORB, BRISK and BRIEF descriptors
        'bf_h2': cv2.NORM_HAMMING2,  # For ORB when WTA_K==3 or 4
        'bf_sl2': cv2.NORM_L2SQR,
    }

    matcher_type = ''
    matcher = None
    extended = False
    bf_cross_check = False

    def __init__(self, matcher_type, *, extended=False, bf_cross_check=False):
        """ Feature matching algorithms.

        They require two sets of descriptors. The first one is from the
        original image, the second one is from the testing image.

        Parameters:
        :matcher_type str: - A type of the matcher to be used. One of the above.
        """
        self.matcher_type = matcher_type.lower()
        self.extended = extended
        self.bf_cross_check = bf_cross_check
        self.matcher = self._get_matcher()

    def _get_matcher(self):
        # https://docs.opencv.org/4.11.0/db/d39/classcv_1_1DescriptorMatcher.html
        # https://docs.opencv.org/4.11.0/d5/d51/group__features2d__main.html
        match self.matcher_type.lower():
            case '' | 'bf_l2':
                if self.extended:
                    matcher = cv2.BFMatcher.create(
                        cv2.NORM_L2,
                        self.bf_cross_check,
                    )
                else:
                    matcher = cv2.DescriptorMatcher.create(
                        cv2.DescriptorMatcher_BRUTEFORCE)
            case 'bf_l1':
                if self.extended:
                    matcher = cv2.BFMatcher.create(
                        cv2.NORM_L1,
                        self.bf_cross_check,
                    )
                else:
                    matcher = cv2.DescriptorMatcher.create(
                        cv2.DescriptorMatcher_BRUTEFORCE_L1)
            case 'bf_h':
                if self.extended:
                    matcher = cv2.BFMatcher.create(
                        cv2.NORM_HAMMING,
                        self.bf_cross_check,
                    )
                else:
                    matcher = cv2.DescriptorMatcher.create(
                        cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
            case 'bf_h2':
                if self.extended:
                    matcher = cv2.BFMatcher.create(
                        cv2.NORM_HAMMING2,
                        self.bf_cross_check,
                    )
                else:
                    matcher = cv2.DescriptorMatcher.create(
                        cv2.DescriptorMatcher_BRUTEFORCE_HAMMINGLUT)
            case 'bf_sl2':
                if self.extended:
                    matcher = cv2.BFMatcher.create(
                        cv2.NORM_L2SQR,
                        self.bf_cross_check,
                    )
                else:
                    matcher = cv2.DescriptorMatcher.create(
                        cv2.DescriptorMatcher_BRUTEFORCE_SL2)
            case 'flann':
                if self.extended:
                    matcher = self.flann()
                else:
                    self.extended = True
                    matcher = cv2.DescriptorMatcher.create(
                        cv2.DescriptorMatcher_FLANNBASED)
            case _:
                raise ValueError(
                    'Such a matcher type `{self.matcher_type}` is not implemented. '
                    'Available matcher types are:\n'
                    '\t`bf_l2`,\n'
                    '\t`bf_l1`,\n'
                    '\t`bf_h`,\n'
                    '\t`bf_h2`,\n'
                    '\t`bf_sl2`,\n'
                    '\t`flann`.'
                )

        return matcher

    def match(
        self,
        des1,
        des2,
        knn_distance=0.7,
        k=1,
        min_match_count=0,
        verbose=False,
    ):
        """A matching method for the Brute-Force matcher. """
        if self.extended:
            return self.knn_match(des1, des2, k, knn_distance, min_match_count, verbose)
        else:
            return self.bf_match(des1, des2, min_match_count, verbose)

    def knn_match(self, des1, des2, k, knn_distance, min_match_count, verbose=False):
        """ A type of matcher for both, Brute-Force and FLANN matchers.

        These extended variants of DescriptorMatcher::match methods find
        several best matches for each query descriptor. The matches are
        returned in the distance increasing order. See
        DescriptorMatcher::match for the details about query and train
        descriptors.

        Parameters:
        :queryDescriptors array: - Query set of descriptors.
        :trainDescriptors array: - Train set of descriptors. This set is
            not added to the train descriptors collection stored in the
            class object.
        :mask array=None: - Mask specifying permissible matches between an
            input query and train matrices of descriptors.
        :k int: - Count of best matches found per each query descriptor
            or less if a query descriptor has less than k possible matches
            in total.
        :compactResult bool=False: - Parameter used when the mask (or masks)
            is not empty. If compactResult is false, the matches vector
            has the same size as queryDescriptors rows. If compactResult is
            true, the matches vector does not contain matches for fully
            masked-out query descriptors.

        Returns:
        :matches array: - Matches. Each matches[i] is k or less matches
            for the same query descriptor.
        """
        if not des1.any() or not des2.any():
            return [[], []]

        matches = self.matcher.knnMatch(des1, des2, k)
        matches = [t for t in matches if len(t) == 2]

        good = []
        # Need to draw only good matches, so create a mask
        #matchesMask = [[0, 0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < knn_distance * n.distance:
                #matchesMask[i] = [1, 0]
                # Save points from the reference image.
                good.append(m)

        if verbose:
            print(f'Found {len(good)} matches.')

        return good

    def bf_match(self, des1, des2, min_match_count, verbose=False):
        """ A matching method for the Brute-Force matcher. """
        matches = self.matcher.match(des1, des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key=attrgetter('distance'))
        if verbose:
            print(f'Found {len(matches)} matches.')

        if min_match_count:
            matches = matches[:min_match_count]

        return matches

    def brute_force(self, normType=cv2.NORM_L2, crossCheck=True):
        """ Brute-Force matching algorithm.

        BFMatcher is simple: it takes the descriptor of one feature in
        first set and is matched with all other features in second set
        using some distance calculation. And the closest one is returned.

        Once it is created, two important methods are BFMatcher.match()
        and BFMatcher.knnMatch(). First one returns the best match.
        Second method returns k best matches where k is specified by the
        user. It may be useful when we need to do additional work on that.

        Parameters:
        :normType int=NORM_L2: - One of
                NORM_L1,
                NORM_L2,
                NORM_HAMMING,
                NORM_HAMMING2.
            L1 and L2 norms are preferable choices for SIFT and SURF
            descriptors, NORM_HAMMING should be used with ORB, BRISK
            and BRIEF, NORM_HAMMING2 should be used with ORB when
            WTA_K==3 or 4 (see ORB::ORB constructor description).
        :crossCheck bool=False: - If it is false, this will be the default
            BFMatcher behaviour when it finds the k nearest neighbours for
            each query descriptor. If crossCheck==true, then the knnMatch()
            method with k=1 will only return pairs (i,j) such that for i-th
            query descriptor the j-th descriptor in the matcher's collection
            is the nearest and vice versa, i.e. the BFMatcher will only
            return consistent pairs. Such technique usually produces best
            results with minimal number of outliers when there are
            enough matches. This is alternative to the ratio test, used
            by D. Lowe in SIFT paper.
        """
        # Initiate BFMatcher
        return cv2.BFMatcher.create(normType, crossCheck)

    def flann(
        self,
        index_params: dict=dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,  # 12
            key_size=12,  # 20
            multi_probe_level=1,  # 2
        ),
        search_params: dict=dict(checks=100),
    ):
        """Fast Library for Approximate Nearest Neighbors (FLANN) feature
        matching algorithm.

        It contains a collection of algorithms optimized for fast nearest
        neighbor search in large datasets and for high dimensional features.
        It works faster than BFMatcher for large datasets.

        Parameters:
        :index_params dict: - For algorithms like SIFT, SURF etc. you can pass
            the following:
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

            While using ORB, you can pass the following. The commented values
            are recommended as per the docs, but it didn't provide required
            results in some cases. Other values worked fine:
                FLANN_INDEX_LSH = 6
                index_params = dict(
                    algorithm=FLANN_INDEX_LSH,
                    table_number=6,  # 12
                    key_size=12,  # 20
                    multi_probe_level=1,  # 2
                )

        :search_params dict: - It specifies the number of times the trees
            in the index should be recursively traversed. Higher values give
            better precision, but also take more time. If you want to change
            the value, pass:
            `search_params = dict(checks=100)`.
        """
        #print('Running FLANN matcher ...')
        return cv2.FlannBasedMatcher(index_params, search_params)
