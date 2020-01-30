"""
MIT License

Copyright (c) 2019 Chodera lab // Memorial Sloan Kettering Cancer Center,
Weill Cornell Medical College, and Authors

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


# =============================================================================
# module functions
# =============================================================================
def bond(x, k, l):

    return tf.math.multiply(
        tf.math.multiply(
            0.5,
            k),
        tf.math.square(
            tf.math.subtract(
                x,
                l)))

def angle(x, k, l):
    return tf.math.multiply(
        tf.math.multiply(
            0.5,
            k),
        tf.math.square(
            tf.math.subtract(
                x,
                l)))


def torsion(angle_idxs, coordinates):
    pass

def lj(
        x,
        sigma_pair,
        epsilon_pair,
        switch=0.0,
        damping=0.0):
    """ Calculate the 12-6 Lenard Jones energy.

    """
    sigma_over_r = tf.where(
        tf.greater(
            x,
            switch),
        tf.math.divide_no_nan(
            sigma_pair,
            tf.math.add(
                x,
                damping)),
        tf.zeros_like(x))

    return tf.math.multiply(
            tf.math.multiply(
                0.5,
                epsilon_pair),
            tf.math.subtract(
                tf.math.pow(
                    sigma_over_r,
                    12.0),
                tf.math.pow(
                    sigma_over_r,
                    6.0)))
