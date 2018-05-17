import glob
import imageio
import numpy as np
import os
import tensorflow as tf

import bn_common

def main():

    # Read images from directory (size must be the same) or single input file
    imgs = []
    filenames = []
    file_paths = glob.glob('/Users/ysono/Downloads/lyft_training_data/Train/CameraRGB/*')[:5]
    for file_path in file_paths:
        ext = file_path.split('.')[-1].lower()

        if ext == 'png' or ext == 'jpg':
            img = imageio.imread(file_path)
            imgs.append(img)

            filename = os.path.basename(file_path)
            filenames.append(filename)
    imgs = np.array(imgs)
    print(imgs.shape)

    sess, raw_output_up = bn_common.recreate_bn_model(imgs)
    
    outputs = sess.run(raw_output_up)
    sess.close()

    os.makedirs('ysono-output', exist_ok=True)
    for output, filename in zip(outputs, filenames):
        output_rgb = np.eye(3)[output]
        imageio.imwrite(os.path.join('ysono-output', filename), output_rgb)


if __name__ == '__main__':
    main()
