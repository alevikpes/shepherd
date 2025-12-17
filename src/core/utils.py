import datetime
from pathlib import Path

import cv2
import numpy as np


class Colours16:

    START_MSG = '\033['
    END_MSG = f'{START_MSG}0m'
    BOLD = '1'
    FG_RED = '31'
    FG_GREEN = '32'
    BG_BLACK = '40m'
    RED_ON_BLACK_BOLD = f'{START_MSG};{BOLD};{FG_RED};{BG_BLACK}'
    GREEN_ON_BLACK_BOLD = f'{START_MSG};{BOLD};{FG_GREEN};{BG_BLACK}'

    @classmethod
    def coloured_text(cls, text, font_style, fg_colour, bg_colour):
        print(
            f'{cls.START_MSG};'
            f'{font_style};'
            f'{fg_colour};'
            f'{bg_colour}'
            f'{text}'
            f'{cls.END_MSG}'
        )


def close_all_windows(win_name: str, wait_time: int=1000) -> None:
    def is_win_visible():
        is_vis = cv2.getWindowProperty(
            win_name,
            cv2.WND_PROP_VISIBLE,
        )
        return is_vis

    # Stopped working for some reason. Always shows -1.
    #while is_win_visible() >= 1:
    while True:
        k = cv2.waitKey(wait_time) & 0xFF
        # Close all windows if one of the ESC or `q` buttons pressed.
        if k == 27 or k == ord('q'):
            cv2.destroyAllWindows()
            break


def get_case_name(prj_path):
    if not isinstance(prj_path, Path):
        prj_path = Path(prj_path)

    project_path = str(prj_path.parents[1])
    case_name = str(prj_path.name)
    return project_path, case_name


def is_image_gray(img):
        #if len(img.shape) != 2:
        #    return False

        return True if len(img.shape) == 2 else False


def image_file_to_ndarray(image, colour_scheme):
    if isinstance(image, str) or isinstance(image, Path):
        img = cv2.imread(str(image))
    elif isinstance(image, np.ndarray):
        if colour_scheme == 'gray' and image.shape == 2:
            # If image is gray and the scheme is gray, skip the convertion.
            return image

        img = image
    else:
        raise ValueError(f'This image type is not supported: {type(image)}')

    return img

    #if colour_scheme.lower() == 'gray':
    #    # convert img to gray scale
    #    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #elif colour_scheme.lower() == 'rgb':
    #    # convert frame to RGB
    #    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #elif colour_scheme.lower() == 'hsv':
    #    # convert frame to HSV
    #    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #else:
    #    raise ValueError('Image type can be either `rgb`, `gray` or `hsv`.')


def img2file(img, file_name, verbose=False):
    cv2.imwrite(str(file_name), img)
    if verbose:
        print(f'Image saved into {file_name}.')


def time2seconds(str_time):
    time_obj = datetime.datetime.strptime(str_time, '%H:%M:%S')
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second


#def bulk_matching(ref_img: str, imgs: list, output_path: Path):
#    for i in range(len(imgs)):
#        match_img = imgs[i]
#        result_img = match_imgs(str(ref_img), match_img)
#        filename = output_path / Path(match_img).name
#        img2file(result_img, str(filename))
