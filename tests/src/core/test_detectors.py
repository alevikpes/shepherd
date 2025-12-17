import cv2
import numpy as np
import pytest  # noqa: F401

from shepherd.src.core import detectors


class TestDetectors:

    #def setup(self, mocker):
    #    self.img = mocker.patch(
    #        'src.core.utils.image_file_to_ndarray',
    #        autospec=True,
    #        colour_scheme='gray',
    #    )
    #    self.obj = mocker.patch.object(Enhancer, autospec=True)

    def test_akaze(self, mocker):
        NPClass = mocker.patch('numpy.ndarray', autospec=True)
        np_obj = NPClass.return_value
        np_obj.shape = (5, 5)
        np_obj.dtype = np.uint8

        det = detectors.Detector(np_obj, 'akaze')
        assert isinstance(det.detect(), cv2.AKAZE)
