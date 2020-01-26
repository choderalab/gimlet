import tensorflow as tf

class GraphConv(tf.keras.Model):
    """ Spectral graph convolution.

    https://arxiv.org/pdf/1609.02907.pdf
    """
    def __init__(self, units=64, depth=10):
        super(GraphConv, self).__init__()
        self.d0 = tf.keras.layers.Dense(
            units=units,
            activation='tanh')
        self.d1 = tf.keras.layers.Dense(
            units=units,
            activation='tanh')
        self.d2 = tf.keras.layers.Dense(
            units=units,
            activation='tanh')
        self.depth=depth

    def call(self, atoms, adjacency_map):
        a = tf.math.add(
            adjacency_map,
            tf.transpose(
                adjacency_map))

        a_tilde = tf.math.add(
            a,
            tf.eye(
                tf.shape(a)[0]))

        d_tilde_n_1_2 = tf.linalg.diag(
            tf.math.pow(
                tf.reduce_sum(
                    a_tilde,
                    axis=0),
                tf.constant(
                    -0.5,
                    dtype=tf.float32)))

        x = tf.matmul(
            tf.matmul(
                d_tilde_n_1_2,
                a),
            d_tilde_n_1_2)

        return self.d2(
            tf.matmul(
                x,
                self.d1(
                    tf.matmul(
                        x,
                        self.d0(
                            tf.matmul(
                                x,
                                tf.one_hot(
                                    atoms,
                                    self.depth)))))))
