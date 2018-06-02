"""
This code is based on DrSleep's framework: https://github.com/DrSleep/tensorflow-deeplab-resnet 
"""
from __future__ import print_function

import argparse
import os
import sys
import time

import tensorflow as tf
import numpy as np

from model import ICNet_BN
from tools import decode_labels, prepare_label
from image_reader import ImageReader
from bn_common import extend_3cls_classifier

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

# If you want to apply to other datasets, change following four lines
DATA_DIR = '/'
DATA_LIST_PATH = './list/carla_train_list.txt' 
IGNORE_LABEL = 255 # The class number of background
INPUT_SIZE = '608, 800' # Input size for training

BATCH_SIZE = 16 
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 60001
POWER = 0.9
WEIGHT_DECAY = 0.00001
PRETRAINED_MODEL = './model/icnet_cityscapes_trainval_90k_bnnomerge.npy'
SNAPSHOT_DIR = './snapshots/'
SAVE_PRED_EVERY = 50

# Loss Function = LAMBDA1 * sub4_loss + LAMBDA2 * sub24_loss + LAMBDA3 * sub124_loss
LAMBDA1 = 0.16
LAMBDA2 = 0.4
LAMBDA3 = 1.0

def get_arguments():
    parser = argparse.ArgumentParser(description="ICNet")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--center_crop_size", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict.")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--restore-from", type=str, default=PRETRAINED_MODEL,
                        help="Where restore model parameters from.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--update-mean-var", action="store_true",
                        help="whether to get update_op from tf.Graphic_Keys")
    parser.add_argument("--train-beta-gamma", action="store_true",
                        help="whether to train beta & gamma in bn layer")
    parser.add_argument("--filter-scale", type=int, default=1,
                        help="1 for using pruned model, while 2 for using non-pruned model.",
                        choices=[1, 2])
    parser.add_argument("--loss_mult_nonego_car", type=float, default=6.0)
    parser.add_argument("--loss_mult_road", type=float, default=3.0)
    return parser.parse_args()

def save(saver, sess, logdir, step):
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)
    
   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def get_mask(gt, num_classes, ignore_label):
    less_equal_class = tf.less_equal(gt, num_classes-1)
    not_equal_ignore = tf.not_equal(gt, ignore_label)
    mask = tf.logical_and(less_equal_class, not_equal_ignore)
    indices = tf.squeeze(tf.where(mask), 1)

    return indices

def create_loss(output, label, num_classes, args):
    with tf.variable_scope('optimizer_fscore'):
        logits = tf.sigmoid(output)

        label = prepare_label(label, tf.stack(output.get_shape()[1:3]), num_classes=num_classes, one_hot=True)

        logits_cls0, logits_cls1, logits_cls2 = tf.split(logits, axis=-1, num_or_size_splits=3)
        labels_cls0, labels_cls1, labels_cls2 = tf.split(label, axis=-1, num_or_size_splits=3)

        def f_score(logits_1cls, labels_1cls, beta):
            true_positive = tf.reduce_sum(tf.multiply(logits_1cls, labels_1cls))
            false_positive = tf.reduce_sum(tf.multiply(logits_1cls, (1 - labels_1cls)))
            false_negative = tf.reduce_sum(tf.multiply((1 - logits_1cls), labels_1cls))

            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)

            f = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
            return f

        def correctness(logits, labels):
            return tf.reduce_mean((labels - logits) * labels + logits * (1 - labels))

        f_car = f_score(logits_cls2, labels_cls2, 2.0)
        f_car_loss = 1.0 - f_car

        f_road = f_score(logits_cls1, labels_cls1, 0.5)
        f_road_loss = 1.0 - f_road

        loss_correctness = correctness(logits, label)

        overall_loss = f_car_loss * args.loss_mult_nonego_car + f_road_loss * args.loss_mult_road + loss_correctness

    return overall_loss

def main():
    """Create the model and start the training."""
    args = get_arguments()
    
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    if args.center_crop_size is None:
        center_crop_size = None
    else:
        hc, wc = map(int, args.center_crop_size.split(','))
        center_crop_size = (hc, wc)

    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            DATA_DIR,
            DATA_LIST_PATH,
            input_size,
            center_crop_size,
            args.random_scale,
            args.random_mirror,
            args.ignore_label,
            IMG_MEAN)
        image_batch, label_batch = reader.dequeue(args.batch_size)

    net = ICNet_BN({'data': image_batch}, is_training=True, num_classes=args.num_classes, filter_scale=args.filter_scale)

    sub4_3cls, sub24_3cls, sub124_3cls = extend_3cls_classifier(net)

    num_reclassified_classes = 3

    restore_var = tf.global_variables()
    all_trainable = [v for v in tf.trainable_variables() if ('beta' not in v.name and 'gamma' not in v.name) or args.train_beta_gamma]
   
    loss_sub4 = create_loss(sub4_3cls, label_batch, num_reclassified_classes, args)
    loss_sub24 = create_loss(sub24_3cls, label_batch, num_reclassified_classes, args)
    loss_sub124 = create_loss(sub124_3cls, label_batch, num_reclassified_classes, args)
    
    l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables()
                 if ('weights' in v.name) or ('kernel' in v.name)]
    
    reduced_loss = LAMBDA1 * loss_sub4 +  LAMBDA2 * loss_sub24 + LAMBDA3 * loss_sub124 + tf.add_n(l2_losses)

    # print(tf.get_variable_scope().name)
    # print(','.join([v.__op.original_name_scope for v in l2_losses]))
    # print(','.join([v for v in tf.trainable_variables() if ('beta' in v.name or 'gamma' in v.name)]))
    # tf.summary.FileWriter('./summary', tf.get_default_graph())
    # exit(0)

    # Using Poly learning rate policy 
    base_lr = tf.constant(args.learning_rate)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))
    
    # Gets moving_mean and moving_variance update operations from tf.GraphKeys.UPDATE_OPS
    if args.update_mean_var == False:
        update_ops = None
    else:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        opt_conv = tf.train.MomentumOptimizer(learning_rate, args.momentum)
        grads = tf.gradients(reduced_loss, all_trainable)
        train_op = opt_conv.apply_gradients(zip(grads, all_trainable))

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    
    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=99)

    ckpt = tf.train.get_checkpoint_state(args.snapshot_dir)
    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('Restore from pre-trained model...')
        net.load(args.restore_from, sess)

    # Start queue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    for step in range(args.num_steps):
        start_time = time.time()
        
        feed_dict = {step_ph: step}
        if step % args.save_pred_every == 0:
            loss_value, loss1, loss2, loss3, _ = sess.run([reduced_loss, loss_sub4, loss_sub24, loss_sub124, train_op], feed_dict=feed_dict)
            save(saver, sess, args.snapshot_dir, step)
        else:
            loss_value, loss1, loss2, loss3, _ = sess.run([reduced_loss, loss_sub4, loss_sub24, loss_sub124, train_op], feed_dict=feed_dict)
        duration = time.time() - start_time
        print('step {:d} \t total loss = {:.3f}, sub4 = {:.3f}, sub24 = {:.3f}, sub124 = {:.3f} ({:.3f} sec/step)'.format(step, loss_value, loss1, loss2, loss3, duration))
        
    coord.request_stop()
    coord.join(threads)

    sess.close()

if __name__ == '__main__':
    main()
