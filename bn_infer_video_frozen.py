import numpy as np
import tensorflow as tf

from PIL import Image
import skvideo.io

import base64
from io import BytesIO
import json

import sys

NOTEGO_MASK = np.invert(np.load('udacity-lyft-challenge-ego-mask-bool.npy'))

ROAD_THRESHOLD = 0.8
CAR_THRESHOLD = 0.7

def extract_2cls(logits_2positiveclasses):
    preds_2cls = []
    for logits in logits_2positiveclasses:
        road = (logits[..., 0] > ROAD_THRESHOLD) & NOTEGO_MASK
        nonego_car = (logits[..., 1] > CAR_THRESHOLD) & NOTEGO_MASK
        preds_2cls.append((nonego_car, road))
    return preds_2cls

def encode(array):
    pil_img = Image.fromarray(array.astype(np.uint8))
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def main(frozen_model_path, video_path):
  
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(frozen_model_path, "rb") as f:
        graph_def.ParseFromString(f.read())

    inpt, outpt = tf.import_graph_def(graph_def,
        return_elements=['inputs_600_800:0', 'output_2positiveclasses:0'],
        name='')

    logits = tf.sigmoid(outpt)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    frames = skvideo.io.vread(video_path)
    # frames = frames[:19] # just debug           # TODO remove!!
    # print('shape of all frames', frames.shape, file=sys.stderr)
    
    answer = {}
    frame_idx = 0 # Frame numbering starts at 1
    batch_size = 112

    for batch_i in range(0, frames.shape[0], batch_size):
        frames_batch = frames[batch_i : batch_i+batch_size]

        logits_2positiveclasses = sess.run(logits, feed_dict={inpt: frames_batch})

        preds_2cls = extract_2cls(logits_2positiveclasses)

        for nonego_car, road in preds_2cls:
            frame_idx += 1
            answer[frame_idx] = [encode(nonego_car), encode(road)]

    sess.close()

    print(json.dumps(answer))


if __name__ == '__main__':
    main(sys.argv[-2], sys.argv[-1])
