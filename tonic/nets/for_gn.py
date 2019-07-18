"""
for_gn.py

Here we present the common featurization functions, initialization functions,
and update and aggregation functions for graph nets.

MIT License

Copyright (c) 2019 Chodera lab // Memorial Sloan Kettering Cancer Center,
Weill Cornell Medical College, Nicea Research, and Authors

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
# tf.enable_eager_execution()

# ===========================================================================
# module functions
# ===========================================================================
def symmetry_specified(self, x, bond_order):
    """ Specify the symmetry of the bond and then calculate seperately,
    before concatenating them together.

    """
    return tf.cond(
        lambda: tf.greater(
            bond_order,
            tf.constant(1, dtype=tf.float32)),

        tf.concat(
            [
                sigma_fn(x),
                pi_fn(x)
            ]),

        self.sigma_fn(x))

# ===========================================================================
# module classes
# ===========================================================================
class ConcatenateThenFullyConnect(tf.keras.Model):
    """ Project all the input to the same dimension and then concat, followed
    by subsequent fully connected layers.

    """
    def __init__(self, config):
        super(ConcatenateThenFullyConnect, self).__init__()
        self.config = config
        self.flow = []
        self.is_virgin = True

    def _build(self, n_vars):
        """ Build the network.

        Note that this function is called when the first data point is
        passed on.

        Parameters
        ----------
        n_vars : int,
            number of variables taken in this layer.

        """
        self.identity = (lambda x: x)

        # concatenation step
        for idx in range(n_vars):
            # put the layer as attribute of the model
            setattr(
                self,
                'D_0_%s' % idx,
                tf.keras.layers.Dense(self.config[0]))

            # NOTE:
            # here we don't put the name into the workflow
            # since we already explicitly expressed the flow
            # self.flow.append('D_0_%s' % idx)

        # the rest of the flow
        for idx in range(1, len(self.config)):
            if isinstance(self.config[idx], int):
                # put the layer as attribute of the model
                setattr(
                    self,
                    'D_%s' % idx,
                    tf.keras.layers.Dense(self.config[idx]))

                # put the name into the workflow
                self.flow.append('D_%s' % idx)

            elif isinstance(self.config[idx], float):
                assert self.config[idx] < 1

                # put the layer as attribute of the model
                setattr(
                    self,
                    'O_%s' % idx,
                    tf.keras.layers.Dense(self.config[idx]))

                # put the name into the workflow
                self.flow.append('O_%s' % idx)

            elif isinstance(self.config[idx], str):
                # put the layer as attribute of the model
                activation = self.config[idx]

                if activation == 'tanh':
                    setattr(
                        self,
                        'A_%s' % idx,
                        tf.tanh)

                elif activation == 'relu':
                    setattr(
                        self,
                        'A_%s' % idx,
                        tf.nn.relu)

                elif activation == 'sigmoid':
                    setattr(
                        self,
                        'A_%s' % idx,
                        tf.sigmoid)

                elif activation == 'leaky_relu':
                    setattr(
                        self,
                        'A_%s' % idx,
                        tf.nn.leaky_relu)

                elif activation == 'elu':
                    setattr(
                        self,
                        'A_%s' % idx,
                        tf.nn.elu)

                elif activation == 'softmax':
                    setattr(
                        self,
                        'A_%s' % idx,
                        tf.nn.softmax)

                self.flow.append('A_%s' % idx)

    # @tf.function
    def _call(self, *args):
        # NOTE: not sure why all the args were rendered as a tuple
        #       I think this is something with tensorflow.

        args = args[0]

        x = tf.concat(
            # list of the projected first layer input
            [getattr(self, 'D_0_%s' % idx)(args[idx]) for idx in range(
                self.n_vars)],
            axis=-1) # note that here we concat it at the last dimension

        for fn in self.flow:
            x = getattr(self, fn)(x)

        return x

    def call(self, *args):
        # build the graph if this is the first time this is called
        if self.is_virgin:
            n_vars = int(len(args))
            self.n_vars = n_vars
            self._build(n_vars)
            self.is_virgin = False

        return self._call(args)

    def switch(self, to_test=True):
        if to_test == True:
            for idx, name in enumerate(self.workflow):
                if name.startswith('O'):
                    setattr(self, name, 'identity')

        else:
            for idx, value in enumerate(self.config):
                if isinstance(value, float):
                    setattr(
                        self,
                        'O_%s' % value,
                        tf.keras.layers.Dropout(value))
