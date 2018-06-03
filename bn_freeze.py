def freeze(is_training):
    import sys
    import tensorflow as tf

    import bn_common
    
    tf.reset_default_graph()

    raw_inputs = tf.placeholder(dtype=tf.uint8, shape=(None, 600, 800, 3),
                                name='inputs_600_800')
#     raw_inputs = tf.placeholder(dtype=tf.uint8, shape=(None, 192, 256, 3),
#                                 name='inputs_192_256')

    sess, raw_output_up = bn_common.recreate_bn_model(raw_inputs, is_training=is_training)
#     sess, raw_output_up = bn_common.recreate_bn_model(raw_inputs, is_training=is_training, crop_size=None)

    print('raw_output_up.name', raw_output_up.name)
    
    graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names=['output_2positiveclasses']
    )

    with tf.gfile.FastGFile('./frozen_inference_graph.pb', 'wb') as f:
        f.write(graph_def.SerializeToString())

    sess.close()

freeze(True)