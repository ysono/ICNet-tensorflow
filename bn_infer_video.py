import numpy as np
import tensorflow as tf

from PIL import Image
import skvideo.io

import base64
from io import BytesIO
import json

import sys

import bn_common

def extract_2cls(preds_3cls):
    preds_2cls = []
    for pred_3cls in preds_3cls:
        road = pred_3cls == 1
        nonego_car = pred_3cls == 2
        preds_2cls.append((nonego_car, road))
    return preds_2cls

def encode(array):
    pil_img = Image.fromarray(array.astype(np.uint8))
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def main(video_path):

    frames = skvideo.io.vread(video_path)
    # frames = frames[:19] # just debug           # TODO remove!!
    # with open('./tempinfo.txt', 'a') as f:
    #     f.write('shape of all frames {}'.format(frames.shape))
    raw_inputs = tf.placeholder(dtype=tf.uint8, shape=(None, frames.shape[1], frames.shape[2], 3))

    sess, raw_output_up = bn_common.recreate_bn_model(raw_inputs)
    
    answer = {}
    frame_idx = 0 # Frame numbering starts at 1
    batch_size = 16

    for batch_i in range(0, frames.shape[0], batch_size):
        frames_batch = frames[batch_i : batch_i+batch_size]

        preds_3cls = sess.run(raw_output_up, feed_dict={raw_inputs: frames_batch})

        preds_2cls = extract_2cls(preds_3cls)

        for nonego_car, road in preds_2cls:
            frame_idx += 1
            answer[frame_idx] = [encode(nonego_car), encode(road)]

    sess.close()

    print(json.dumps(answer))


if __name__ == '__main__':
    main(sys.argv[-1])

# '/Users/ysono/code/ysono/_carnd/CarND-T3P2L1-Object-Detection/driving.mp4'
