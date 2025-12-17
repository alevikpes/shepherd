# https://thepythoncode.com/article/extract-frames-from-videos-in-python
# https://techtutorialsx.com/2021/04/29/python-opencv-splitting-video-frames/


import os
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np


SAVING_FRAMES_PER_SECOND = 1
DATA_PATH = Path(__file__).parent.parent.parent / 'data'


class FrameExtractor:

    def __init__(self, video_path, ref_img):
        self.video_path = video_path
        self.ref_img = ref_img

    def _format_timedelta(self, td):
        """Utility function to format timedelta objects in a cool way
        (e.g 00:00:20.05) omitting microseconds and retaining milliseconds.
        """
        result = str(td)
        try:
            result, ms = result.split('.')
        except ValueError:
            return (result + '.00').replace(':', '-')

        ms = int(ms)
        ms = round(ms / 1e4)
        return f'{result}.{ms:02}'.replace(':', '-')

    def _get_saving_frames_durations(self, cap, saving_fps):
        """A function that returns the list of durations where to save
        the frames.
        """
        s = []
        # get the clip duration by dividing number of frames
        # by the number of frames per second
        clip_duration = (
            cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
        # use np.arange() to make floating-point steps
        for i in np.arange(0, clip_duration, 1 / saving_fps):
            s.append(i)

        return s

    def match_frame(self, frame):
        matcher = matcher.Matcher(self.ref_img)
        result_img = matcher.match_imgs(str(match_img))

    def extract_frames(self):
        video_file = str(video_path)
        filename, _ = os.path.splitext(video_file)
        filename += '-opencv'
        print(filename)
        # make a folder by the name of the video file
        if not os.path.isdir(filename):
            os.mkdir(filename)

        # read the video file    
        cap = cv2.VideoCapture(video_file)
        # get the FPS of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
        saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
        # get the list of duration spots to save
        saving_frames_durations = self._get_saving_frames_durations(cap, saving_frames_per_second)
        # start the loop
        count = 0
        while True:
            is_read, frame = cap.read()
            if not is_read:
                # break out of the loop if there are no frames to read
                break
            # get the duration by dividing the frame count by the FPS
            frame_duration = count / fps
            try:
                # get the earliest duration to save
                closest_duration = saving_frames_durations[0]
            except IndexError:
                # the list is empty, all duration frames were saved
                break

            if frame_duration >= closest_duration:
                # if closest duration is less than or equals the frame duration, 
                # then save the frame
                frame_duration_formatted = self._format_timedelta(timedelta(seconds=frame_duration))
                cv2.imwrite(os.path.join(filename, f'frame{frame_duration_formatted}.jpg'), frame) 
                # drop the duration spot from the list, since this duration spot is already saved
                try:
                    saving_frames_durations.pop(0)
                except IndexError:
                    pass

            # increment the frame count
            count += 1

    # auto generated
    def replace_frame(video_path, frame_to_replace, new_frame):
        """
        Replace one frame in a video with another frame.
    
        Args:
        video_path (str): Path to the video file.
        frame_to_replace (int): Index of the frame to replace in the video.
        new_frame (numpy.ndarray): New frame to replace the existing frame.
    
        Returns:
        bool: True if frame replacement was successful, False otherwise.
        """
        # Read the video
        video_capture = cv2.VideoCapture(video_path)
    
        # Check if the video opened successfully
        if not video_capture.isOpened():
            print("Error: Could not open video.")
            return False
    
        frame_count = 0
        success = True
    
        # Loop through each frame in the video
        while success:
            success, frame = video_capture.read()
    
            # Replace the frame at the specified index
            if frame_count == frame_to_replace:
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                video_capture.write(new_frame)
    
            frame_count += 1
    
        video_capture.release()
        return True
    
    ## Example usage
    #video_path = "input_video.mp4"
    #frame_to_replace = 100
    #new_frame = cv2.imread("new_frame.jpg")
    #
    #replace_frame(video_path, frame_to_replace, new_frame)


def find_homography(src_pts, dest_pts, ref_img, match_img):
    print(src_pts)
    print(dest_pts)
    src_pts = np.float32(src_pts)  #.reshape(-1, 1, 2)
    dest_pts = np.float32(dest_pts)  #.reshape(-1, 1, 2)
    print(src_pts)
    print(dest_pts)
    M, mask = cv2.findHomography(src_pts, dest_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w = ref_img.shape
    pts = np.float32(
        [[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]
    ).reshape(-1, 1, 2)
    if pts.any() and M is not None:
        # Draw a matching polygon
        dst = cv2.perspectiveTransform(pts, M)
        match_img = cv2.polylines(
            match_img,
            [np.int32(dst)],
            True,
            (255, 0, 0),
            3,
            cv2.LINE_AA,
        )
        return match_img

    return

def save_video():
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid_out = cv2.VideoWriter('features_vid.avi', fourcc, 20.0, (640, 480))
