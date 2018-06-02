import numpy as np
import tensorflow as tf

from PIL import Image
import skvideo.io

import base64
from io import BytesIO
import json

import sys

NOTEGO_MASK = np.invert(np.load('udacity-lyft-challenge-ego-mask-bool.npy'))

def extract_2cls(preds_3cls):
    preds_2cls = []
    for pred_3cls in preds_3cls:
        road = (pred_3cls == 1) & NOTEGO_MASK
        nonego_car = (pred_3cls == 2) & NOTEGO_MASK
        preds_2cls.append((nonego_car, road))
    return preds_2cls

def extract_2cls_192256(preds_3cls):
    preds_2cls = []
    for pred_3cls in preds_3cls:
        road = (pred_3cls == 1)
        nonego_car = (pred_3cls == 2)
        preds_2cls.append((nonego_car, road))
    return preds_2cls

def encode(array):
    pil_img = Image.fromarray(array.astype(np.uint8))
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def main(frozen_model_path, frozen_model_path_centercrop, video_path):
  
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(frozen_model_path, "rb") as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def)
    inpt, outpt = [
        tf.get_default_graph().get_tensor_by_name(n) for n in [
            'import/inputs_600_800:0', 'import/output_sparse:0'
        ]
    ]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    frames = skvideo.io.vread(video_path)
    # print('shape of all frames', frames.shape, file=sys.stderr)
    
    answer = {}
    frame_idx = 0 # Frame numbering starts at 1
    batch_size = 112

    for batch_i in range(0, frames.shape[0], batch_size):
        frames_batch = frames[batch_i : batch_i+batch_size]

        preds_3cls = sess.run(outpt, feed_dict={inpt: frames_batch})

        preds_2cls = extract_2cls(preds_3cls)

        for nonego_car, road in preds_2cls:
            frame_idx += 1
            answer[frame_idx] = [nonego_car, road]

    sess.close()

    

    tf.reset_default_graph()
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(frozen_model_path_centercrop, "rb") as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def)
    inpt, outpt = [
        tf.get_default_graph().get_tensor_by_name(n) for n in [
            'import/inputs_192_256:0', 'import/output_sparse:0'
        ]
    ]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    center_h, center_w = (192, 256)
    center_y0, center_x0 = (int((608 - 192)/ 2), int((800 - 256) / 2))
    center_y1 = center_y0 + 192
    center_x1 = center_x0 + 256

    frame_idx = 0 # Frame numbering starts at 1
    batch_size = 112
    for batch_i in range(0, frames.shape[0], batch_size):
        frames_batch = frames[batch_i : batch_i+batch_size]
        frames_batch = frames_batch[:, center_y0:center_y1, center_x0:center_x1, :]

        preds_3cls = sess.run(outpt, feed_dict={inpt: frames_batch})

        preds_2cls = extract_2cls_192256(preds_3cls)

        for nonego_car, road in preds_2cls:
            frame_idx += 1

            answer[frame_idx][0][center_y0:center_y1, center_x0:center_x1] = nonego_car
            # answer[frame_idx][0][center_y0:center_y1, center_x0:center_x1] = answer[frame_idx][0][center_y0:center_y1, center_x0:center_x1] | nonego_car
            # answer[frame_idx][0][center_y0:center_y1, center_x0:center_x1] = answer[frame_idx][0][center_y0:center_y1, center_x0:center_x1] & nonego_car

            answer[frame_idx][1][center_y0:center_y1, center_x0:center_x1] = road
            # answer[frame_idx][1][center_y0:center_y1, center_x0:center_x1] = answer[frame_idx][1][center_y0:center_y1, center_x0:center_x1] | road
            # answer[frame_idx][1][center_y0:center_y1, center_x0:center_x1] = answer[frame_idx][1][center_y0:center_y1, center_x0:center_x1] & road

            answer[frame_idx] = [encode(answer[frame_idx][0]), encode(answer[frame_idx][1])]

    sess.close()

    print(json.dumps(answer))


if __name__ == '__main__':
    main(sys.argv[-3], sys.argv[-2], sys.argv[-1])
