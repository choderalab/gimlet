import tensorflow as tf
def is_new(walk):
    # (n_batch, n_atoms)
    is_virgin = tf.constant(
        True,
        shape=(
            tf.shape(walk)[0],
            tf.shape(tf.unique(walk[0])[0])[0]))

    # (n_batch, n_walks)
    is_new_ = tf.constant(
        False,
        shape=tf.shape(walk))

    for idx in range(tf.shape(walk)[1]):
        walk_row = walk[:, idx]

        walk_row_is_virgin = tf.gather_nd(
            is_virgin,
            tf.stack(
                [
                    tf.range(
                        tf.shape(walk_row, tf.int64)[0]),
                    walk_row

                ],
                axis=1))

        virgin_idxs = tf.boolean_mask(
            tf.stack(
                [
                    tf.range(
                        tf.shape(walk_row, tf.int64)[0]),
                    walk_row

                ],
                axis=1),
            walk_row_is_virgin)



        is_virgin = tf.tensor_scatter_nd_update(
            is_virgin,
            virgin_idxs,
            tf.constant(
                False,
                shape=(
                    tf.shape(virgin_idxs)[0],)))


        is_new_ = tf.transpose(
            tf.tensor_scatter_nd_update(
                tf.transpose(is_new_),
                [[idx]],
                tf.expand_dims(walk_row_is_virgin, 0)))

    return is_new_
