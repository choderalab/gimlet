"""
attention.py

MIT License

Copyright (c) 2019 Chodera lab // Memorial Sloan Kettering Cancer Center
and Authors

Authors:
Yuanqing Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# ===========================================================================
# imports
# ===========================================================================
import tensorflow as tf
tf.enable_eager_execution()

# ===========================================================================
# module classes
# ===========================================================================
class Attention(tf.keras.Model):
    """
    multi-head attention mechanism
    adopted from tensorflow.models

    Parameters
    ----------
    hidden_size : int
        size of hidden layer
    num_heads : int
        number of attention heads

    Returns
    -------
    attention_output
    """
    def __init__(self, hidden_size, num_heads):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # query
        self.d_q = tf.keras.layers.Dense(hidden_size, use_bias=False)

        # key
        self.d_k = tf.keras.layers.Dense(hidden_size, use_bias=False)

        # value
        self.d_v = tf.keras.layers.Dense(hidden_size, use_bias=False)

        # output
        self.d_output = tf.layers.Dense(hidden_size, use_bias=False)

    def split_heads(self, x):
        """
        split attention heads

        """
        # get batch size and length
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]

        # Calculate depth of last dimension after it has been split.
        depth = (self.hidden_size // self.num_heads)

        # Split the last dimension
        x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

        # Transpose the result
        return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        """
        combine attention heads

        """
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[2]
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [batch_size, length, self.hidden_size])

    @tf.contrib.eager.defun
    def _call(self, x, y):
        q = self.d_q(x)
        k = self.d_k(y)
        v = self.d_v(y)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Scale q to prevent the dot product between q and k from growing too large.
        depth = (self.hidden_size // self.num_heads)
        q *= depth ** -0.5

        # Calculate dot product attention
        logits = tf.matmul(q, k, transpose_b=True)
        weights = tf.nn.softmax(logits)
        attention_output = tf.matmul(weights, v)

        # Recombine heads --> [batch_size, length, hidden_size]
        attention_output = self.combine_heads(attention_output)

        # Run the combined outputs through another linear projection layer.
        attention_output = self.d_output(attention_output)
        return attention_output

    def __call__(self, x, y):
        return self._call(x, y)
