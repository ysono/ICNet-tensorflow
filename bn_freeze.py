

def main(is_training):
    import sys
    import tensorflow as tf

    import bn_common

    raw_inputs = tf.placeholder(dtype=tf.uint8, shape=(None, 600, 800, 3),
                                name='inputs_600_800')


    sess, raw_output_up = bn_common.recreate_bn_model(raw_inputs, is_training=is_training)

    # tf.summary.FileWriter('./summary', tf.get_default_graph())
    # sess.close()
    # exit(0)

    print('raw_output_up.name', raw_output_up.name)
    
    graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names=['output_sparse']
    )

    with tf.gfile.FastGFile('./frozen_inference_graph.pb', 'wb') as f:
        f.write(graph_def.SerializeToString())

    sess.close()

if __name__ == '__main__':
    is_training = len(sys.argv) >= 2 and sys.argv[1] == 'is_training'
    main(is_training)
