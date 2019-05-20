"""
gp.py

Bayesian Optimization Using Gaussian Process.

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

# =============================================================================
# imports
# =============================================================================
import tensorflow as tf
tf.enable_eager_execution()
import random

# =============================================================================
# module functions
# =============================================================================
def optimize(
        object_fn,
        space,
        n_calls,
        minimize=True):
    """ Random grid search.

    """
    idx = 0
    xs = []
    ys = []

    while idx < n_calls:
        x = [random.choice(dimension) for dimension in space]
        y = object_fn(x)
        xs.append(x)
        ys.append(y)

    xs = tf.convert_to_tensor(xs)
    ys = tf.convert_to_tensor(ys)
    ys = tf.reshape(ys, [-1])

    y_max = tf.math.maximum(ys)
    max_idx = tf.argsort(ys)
    x_max = xs[idx]

    return x_max, y_max
