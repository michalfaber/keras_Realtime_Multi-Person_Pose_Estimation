import os
import sys
import argparse
import cv2
import time
from config_reader import config_reader

from processing import extract_parts, draw

from model.cmu_model import get_testing_model

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

currentDT = time.localtime()
start_datetime = time.strftime("-%m-%d-%H-%M-%S", currentDT)


def crop(image, w, f):
    return image[:, int(w * f): int(w * (1 - f))]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0, help='ID of the device to open')
    parser.add_argument('--model', type=str, default='model/keras/model.h5', help='path to the weights file')
    parser.add_argument('--frame_ratio', type=int, default=7, help='analyze every [n] frames')
    # --process_speed changes at how many times the model analyzes each frame at a different scale
    parser.add_argument('--process_speed', type=int, default=1,
                        help='Int 1 (fastest, lowest quality) to 4 (slowest, highest quality)')
    parser.add_argument('--out_name', type=str, default=None, help='name of the output file to write')
    parser.add_argument('--mirror', type=bool, default=True, help='whether to mirror the camera')

    args = parser.parse_args()
    device = args.device
    keras_weights_file = args.model
    frame_rate_ratio = args.frame_ratio
    process_speed = args.process_speed
    out_name = args.out_name
    mirror = args.mirror

    print('start processing...')

    # load model
    # authors of original model don't use
    # vgg normalization (subtracting mean) on input images
    model = get_testing_model()
    model.load_weights(keras_weights_file)

    # load config
    params, model_params = config_reader()

    # Video reader
    cam = cv2.VideoCapture(device)
    # CV_CAP_PROP_FPS
    input_fps = cam.get(cv2.CAP_PROP_FPS)
    print("Running at {} fps.".format(input_fps))

    ret_val, orig_image = cam.read()

    width = orig_image.shape[1]
    height = orig_image.shape[0]
    factor = 0.3

    out = None
    # Output location
    if out_name is not None and ret_val is not None:
        output_path = 'videos/outputs/'
        output_format = '.mp4'
        video_output = output_path + out_name + output_format

        # Video writer
        output_fps = input_fps / frame_rate_ratio

        tmp = crop(orig_image, width, factor)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_output, fourcc, output_fps, (tmp.shape[1], tmp.shape[0]))

        del tmp

    scale_search = [0.22, 0.25, .5, 1, 1.5, 2]  # [.5, 1, 1.5, 2]
    scale_search = scale_search[0:process_speed]

    params['scale_search'] = scale_search

    i = 0  # default is 0
    resize_fac = 8
    # while(cam.isOpened()) and ret_val is True:
    while True:

        cv2.waitKey(10)

        if cam.isOpened() is False or ret_val is False:
            break

        if mirror:
            orig_image = cv2.flip(orig_image, 1)

        tic = time.time()

        cropped = crop(orig_image, width, factor)

        input_image = cv2.resize(cropped, (0, 0), fx=1/resize_fac, fy=1/resize_fac, interpolation=cv2.INTER_CUBIC)

        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

        # generate image with body parts
        all_peaks, subset, candidate = extract_parts(input_image, params, model, model_params)
        canvas = draw(cropped, all_peaks, subset, candidate, resize_fac=resize_fac)

        print('Processing frame: ', i)
        toc = time.time()
        print('processing time is %.5f' % (toc - tic))

        if out is not None:
            out.write(canvas)

        # canvas = cv2.resize(canvas, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        cv2.imshow('frame', canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        ret_val, orig_image = cam.read()

        i += 1
