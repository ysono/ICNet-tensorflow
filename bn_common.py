import numpy as np
import sys
import tensorflow as tf

from model import ICNet_BN

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

def extend_3cls_classifier(net):
    sub4_out, sub24_out, sub124_out, conv2_sub1_bn, conv1_sub1_bn = [net.layers[n] for n in [
        'sub4_out', 'sub24_out', 'conv6_cls', 'conv2_sub1_bn', 'conv1_sub1_bn']]

    with tf.variable_scope('reclassification'):
        num_reclassified_classes = 3

        sub4_3cls, sub24_3cls, sub124_3cls = [
            tf.layers.conv2d(logits_19cls,
                filters=num_reclassified_classes, kernel_size=kernel_size, strides=1, padding='SAME',
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
            for logits_19cls, kernel_size in zip(
                [sub4_out, sub24_out, sub124_out],
                [3, 5, 7]
            )
        ]

        skip_quartersize = 0.0001 * conv2_sub1_bn
        skip_quartersize = tf.layers.conv2d(skip_quartersize,
            filters=num_reclassified_classes, kernel_size=9, strides=1, padding='SAME',
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
        sub124_3cls_interp_to_quartersize = tf.image.resize_bilinear(
            sub124_3cls, size=tf.shape(skip_quartersize)[1:3], align_corners=True)
        sub124_3cls_added_quartersize = sub124_3cls_interp_to_quartersize + skip_quartersize
        
        skip_halfsize = 0.00001 * conv1_sub1_bn
        skip_halfsize = tf.layers.conv2d(skip_halfsize,
            filters=num_reclassified_classes, kernel_size=11, strides=1, padding='SAME',
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
        sub124_3cls_interp_to_halfsize = tf.image.resize_bilinear(
            sub124_3cls_added_quartersize, size=tf.shape(skip_halfsize)[1:3], align_corners=True)
        sub124_3cls_added_halfsize = sub124_3cls_interp_to_halfsize + skip_halfsize

    return sub4_3cls, sub24_3cls, sub124_3cls_added_halfsize

def recreate_bn_model(input_imgs_tensor):
    snapshot_dir = './snapshots/'
    restore_from = './model/icnet_cityscapes_trainval_90k_bnnomerge.npy'

    img_r, img_g, img_b = tf.split(axis=3, num_or_size_splits=3, value=input_imgs_tensor)
    imgs = tf.cast(tf.concat(axis=3, values=[img_b, img_g, img_r]), dtype=tf.float32)
    imgs = imgs - IMG_MEAN

    net = ICNet_BN({'data': imgs}, is_training=True, num_classes=19, filter_scale=1)

    _, _, sub124_3cls = extend_3cls_classifier(net)

    restore_var = tf.global_variables()

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    ckpt = tf.train.get_checkpoint_state(snapshot_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print('restoring from', ckpt.model_checkpoint_path, file=sys.stderr)
        loader = tf.train.Saver(var_list=restore_var)
        loader.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('restoring from', restore_from, file=sys.stderr)
        sess.run(tf.global_variables_initializer())
        net.load(restore_from, sess)

    # Predictions.
    raw_output_up = tf.image.resize_bilinear(sub124_3cls, size=(608, 800), align_corners=True)
    raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, 600, 800)
    raw_output_up = tf.argmax(raw_output_up, axis=3)

    return sess, raw_output_up
