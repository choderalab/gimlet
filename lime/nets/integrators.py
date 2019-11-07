"""
attention.py

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

import tensorflow as tf

class BAOAB(tf.keras.optimizers.Optimizer):
    """ BAOAB integrator for Bayesian neural networks,
    among other things.

    Reference:
    arXiv: 1908.11843

    A : linear kick
        $$
        p = p + h \gamma t
        $$

    B : linear drift
        $$
        \theta = \theta + h G(\theta)
        $$

    O : Ornstein-Uhlenbeck
        $$
        \alpha p + \sqrt{\tao (1 - \alpha ^ 2)} R_n,

        $$

        where $\alpha = e^{-\gamma * h}$ and $R_n \sim \mathcal{N}(0, 1).

    """
    def __init__(
        self,
        tao=1e-6,
        h=1e-2,
        gamma=10,
        name='BAOAB'):

        super(BAOAB, self).__init__(name)

        # convert the hyperparameters to tensors
        self.tao = tf.convert_to_tensor(tao, dtype=tf.float32)
        self.h = tf.convert_to_tensor(h, dtype=tf.float32)
        self.gamma = tf.convert_to_tensor(gamma, dtype=tf.float32)

    def get_config(self):
        config = super(Adam, self).get_config()
        return config

    def _create_slots(self, var_list):
        """ Prepare the slots for the variables in `var_list`

        Parameters
        ----------
        var_list : list of tf.Variables

        """
        for var in var_list:
            # add a slot for the first-order derivative, i.e. velocity,
            # for each variable
            self.add_slot(
                var,
                'p',)

    def _resource_apply_dense(self, grad, var):
        """ Apply gradient to the dense variable.

        """
        p = self.get_slot(
            var,
            'p')

        # print(p)

        # B
        # $$ p = p + \frac{h}{2}G(\theta_n) $$
        p = tf.math.add(
            p,
            tf.math.multiply(
                tf.multiply(
                    tf.constant(0.5, var.dtype.base_dtype),
                    tf.cast(
                        self.h,
                        var.dtype.base_dtype)),
                -grad))

        # A
        # $$ \theta = \theta + \frac{h}{2} p $$
        theta = tf.math.add(
            var,
            tf.math.multiply(
                tf.multiply(
                    tf.constant(0.5, dtype=var.dtype.base_dtype),
                    tf.cast(
                        self.h,
                        var.dtype.base_dtype)),
                p))

        # O
        # $$ p = \alpha p + \sqrt{\tao (1 - \alpha ^ 2)} R_n $$
        alpha = tf.math.exp(
            tf.math.multiply(
                -self.gamma,
                self.h))

        p = tf.math.add(
                tf.math.multiply(
                    alpha,
                    p),
                tf.math.multiply(
                    tf.math.sqrt(
                        self.tao,
                        tf.math.subtract(
                            tf.constant(1, dtype=var.dtype.base_dtype),
                            tf.math.square(alpha))),
                    tf.random.normal(
                        shape=tf.shape(var))))


        # A
        # $$ \theta = \theta + \frac{h}{2} p $$
        theta = tf.math.add(
            theta,
            tf.math.multiply(
                tf.multiply(
                    tf.constant(0.5, dtype=var.dtype.base_dtype),
                    tf.cast(
                        self.h,
                        var.dtype.base_dtype)),
                p))

        # B
        # $$ p = p + \frac{h}{2}G(\theta_n) $$
        p = tf.math.add(
            p,
            tf.math.multiply(
                tf.multiply(
                    tf.constant(0.5, var.dtype.base_dtype),
                    tf.cast(
                        self.h,
                        var.dtype.base_dtype)),
                -grad))


        var.assign(
            theta)


        self.get_slot(
                var,
                'p').assign(
            p)
