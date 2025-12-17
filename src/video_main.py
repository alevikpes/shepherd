"""
Codecs for videos (fourcc):
* mp4v - for mp4
* xvid - for avi
* h264 - for mp4
* x264
* avc1 - for mp4
"""

import time
from pathlib import Path

import cv2

from shepherd.src.core.utils import image_file_to_ndarray, img2file, time2seconds
from shepherd.src.object_detector import ObjectDetector


if __name__ == '__main__':
    DATA_PATH = Path(__file__).parent.parent / 'data'
    PROJECT_PATH = DATA_PATH / 'bike'
    CASE_NAME = 'case1_rgb'

    # Switches.
    SAVE_OUTPUT = False
    SHOW_FEATURES = True
    MARK_FEATURES = False
    MARK_OBJECT = True
    SHOW_MATCHES = False
    image_size = None
    #image_size=(3600, 1200)

    print(PROJECT_PATH, CASE_NAME)
    od = ObjectDetector(
        project_path=PROJECT_PATH,
        case_name=CASE_NAME,
        case_type='video',
    )

    # Resize ref image.
    cv2.resize(od.ref_img, (320, 240))

    # Find features for the reference image
    kp_ref, des_ref = od.find_features(
        od.ref_img,
    )

    # path to video
    VIDEO_IN = od.test_video_path
    print(VIDEO_IN)
    vid_out = None
    if VIDEO_IN == 'webcam':
        video = cv2.VideoCapture(0)
        # Size of the video frame.
        MIN_X = 0
        MAX_X = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        MIN_Y = 0
        MAX_Y = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'Video resolution: {MAX_X}:{MAX_Y}')

        ## Set focus. NOTE: doesn't work.
        ##focus = 250  # min: 0, max: 255, increment:5
        #video.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        ##video.set(cv2.CAP_PROP_FOCUS, focus)

        start_frame = 0
        start_time_sec = 0
    else:
        video = cv2.VideoCapture(VIDEO_IN)
        fourcc_int = int(video.get(cv2.CAP_PROP_FOURCC))
        #print('fourcc_int: ', fourcc_int)
        fourcc_current = (
            chr((fourcc_int & 0XFF)) +
            chr((fourcc_int & 0XFF00) >> 8) +
            chr((fourcc_int & 0XFF0000) >> 16) +
            chr((fourcc_int & 0XFF000000) >> 24)
            #chr(0)
        )
        print(f'Original codec: {fourcc_current}')

        # start processing from the specified frame number
        start_time = od.test_video_start_time
        end_time = od.test_video_end_time
        print(f'Start time: {start_time}')
        print(f'End time: {end_time}')
        start_time_sec = time2seconds(start_time)
        end_time_sec = time2seconds(end_time)
        print(f'Start time seconds: {start_time_sec}')
        print(f'End time seconds: {end_time_sec}')
        # Set the starting and ending frame numbers
        start_frame = int(orig_fps * start_time_sec)
        end_frame = int(orig_fps * end_time_sec)
        print(f'Start frame: {start_frame}')
        print(f'End frame: {end_frame}')
        print(f'Total frames: {end_frame - start_frame}')

        # Iterate over the frames
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        video.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)

    if SAVE_OUTPUT:
        # Define a codec and create a VideoWriter object
        # for saving video into a file.
        fourcc = cv2.VideoWriter_fourcc(*(od.test_video_codec))
        #fourcc = -1  # Shows all codecs.
        #fourcc = od.test_video_codec
        #print(f'fourcc: {fourcc}')
        frame_width = int(video.get(3))
        frame_height = int(video.get(4))
        frame_size = (frame_width, frame_height)
        orig_fps = int(video.get(cv2.CAP_PROP_FPS))
        print(f'Original video frame size: {frame_width}, {frame_height}')
        print(f'Original video FPS: {orig_fps}')

        vid_out = cv2.VideoWriter(
            VIDEO_OUT,
            fourcc,
            orig_fps,
            frame_size,
            isColor=True,
        )

    #print(f'Current frame: {int(video.get(cv2.CAP_PROP_POS_FRAMES))}')
    #print(f'Current time: {int(video.get(cv2.CAP_PROP_POS_MSEC))}')

    # Initialize variables for FPS calculation
    t0 = time.time()
    n_frames = 0
    avg_fps = 0

    current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))

    video_playing = True
    #while int(video.get(cv2.CAP_PROP_POS_FRAMES)) <= end_frame:
    while video_playing:
        if VIDEO_IN == 'webcam' and video.isOpened():
            video_playing = True
        elif (
            VIDEO_IN != 'webcam' and
            int(video.get(cv2.CAP_PROP_POS_FRAMES)) <= end_frame
        ):
            video_playing = True
        else:
            video_playing = False

        #print(f'\nStart time: {start_time_sec}')
        #print(f'End time: {end_time_sec}')
        #print(f'Start frame: {start_frame}')
        #print(f'End frame: {end_frame}')
        #print(f'Current time: {int(video.get(cv2.CAP_PROP_POS_MSEC))} msec')
        #print(f'Current frame: {int(video.get(cv2.CAP_PROP_POS_FRAMES))}')
        #print(f'Current frame: {current_frame}')

        # reading video
        ret, frame = video.read()

        if ret:
            od.test_img = image_file_to_ndarray(frame, od.colour_scheme)

            ## Save every hundreds frame to file.
            #frame_num = int(video.get(cv2.CAP_PROP_POS_FRAMES))
            #if frame_num % 100 == 0:
            #    filename = (
            #        case.single_input_path / 
            #        f'{CASE_NAME}_{frame_num}.png'
            #    )
            #    img2file(frame, filename)
            #    print(f'Frame #{frame_num} saved into {filename}')

            ### Find features and descriptors for the frame
            kp_test, des_test = od.find_features(od.test_img)

            # Find matches
            src_pts, dst_pts = od.get_good_matches(
                kp_ref,
                des_ref,
                kp_test,
                des_test,
                #k=2,
                verbose=False,
            )

            # Mark features on the frame.
            if SHOW_FEATURES:
                if MARK_FEATURES:
                    od.ref_img = od.draw_features(
                        od.ref_img,
                        src_pts,
                        mark_features=True,
                    )
                    od.test_img = od.draw_features(
                        od.test_img,
                        dst_pts,
                        mark_features=True,
                    )
                else:
                    od.ref_img = od.draw_features(od.ref_img, src_pts)
                    od.test_img = od.draw_features(od.test_img, dst_pts)

            # Draw frame on the frame.
            if MARK_OBJECT:
                od.test_img = od.draw_polygon(od.test_img, src_pts, dst_pts)

            # Draw matches
            if SHOW_MATCHES:
                img = od.draw_matches()
            else:
                img = od.test_img

            # Save frame into the video file
            if SAVE_OUTPUT:
                vid_out.write(img)

            n_frames += 1
            elapsed_time = time.time() - t0
            avg_fps = n_frames / elapsed_time

            # Display text in the frame.
            img = od.display_text(
                img,
                (
                    f'Avg. FPS: {int(avg_fps)}',
                    #f'Features found: {len(od.matches)}',
                    f'Detector: {od.detector_type.upper()}',
                    f'Descriptor: {od.descriptor_type.upper()}',
                    f'Matcher: {od.matcher_type.upper()}',
                ),
            )

            # Play video
            if VIDEO_IN == 'webcam':
                cv2.imshow(f'Ref Image', od.ref_img)
                cv2.imshow(f'{PROJECT_PATH}, {CASE_NAME}', img)
            else:
                cv2.imshow(f'Ref Image', od.ref_img)
                cv2.imshow(f'{Path(VIDEO_IN).name}', od.test_img)

            k = cv2.waitKey(5) & 0xFF
            if k == 27 or k == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            print('Can`t read video')
            break

    print(f'\nAverage FPS: {avg_fps}')
    print(f'Analysed frames: {n_frames}')

    video.release()
    if vid_out:
        print(f'\nVideo saved to: {od.result_file_name}')
        vid_out.release()

    cv2.destroyAllWindows()
