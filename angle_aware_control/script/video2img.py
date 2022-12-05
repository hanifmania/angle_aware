import cv2
import os

def save_all_frames(video_path, dir_path, basename, rate, ext='jpg'):
    print(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        for _ in range(rate):
            ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
            n += 1
        else:
            return

file_dir = os.path.dirname(__file__)
input_video_path = file_dir + '/temp/video.mp4'
output_dir_path = file_dir + "/temp/result"
save_all_frames(input_video_path, output_dir_path,  'img', 1)

# save_all_frames('data/temp/sample_video.mp4', 'data/temp/result_png', 'sample_video_img', 'png')