"""
1d_conv.py

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
# module classes
# ===========================================================================

class OneDConvNet(tf.keras.Model):
    """ Encoder that contains both convolutional and recurrent layer.
    """
    def __init__(
        self,
        workflow_input) -> None:

        # bookkeeping
        super(OneDConvNet, self).__init__()

        # define the workflow
        self.workflow_input = workflow_input
        self.workflow = []

        # run build() once to construct the model
        self._build_layers()

    def _build_layers(self):
        for idx, layer in enumerate(self.workflow_input):
            assert isinstance(layer, str)
            if layer.startswith('C'):
                # get the configs
                config = layer.split('_')
                if len(config) == 3:
                    _, conv_unit, conv_kernel_size = config
                    conv_unit = int(conv_unit)
                    conv_kernel_size = int(conv_kernel_size)

                    # set the attributes
                    name = 'C' + str(idx)
                    self.workflow.append(name)
                    setattr(
                        self,
                        name,
                        tf.keras.layers.Conv1D(
                            conv_unit, conv_kernel_size,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01)))



                elif len(config) == 4:
                    _, conv_unit, conv_kernel_size, dilation = config
                    conv_unit = int(conv_unit)
                    conv_kernel_size = int(conv_kernel_size)
                    dilation = int(dilation)

                    # set the attributes
                    name = 'C' + str(idx)
                    self.workflow.append(name)
                    setattr(
                        self,
                        name,
                        tf.keras.layers.Conv1D(
                            conv_unit, conv_kernel_size,
                            kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                            bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                            dilation_rate=dilation))


            elif layer.startswith('P'):
                # get the configs
                _, pool_size = layer.split('_')
                pool_size = int(pool_size)

                # set the attributes
                name = 'P' + str(idx)
                # pool layers
                if pool_size > 0:
                    self.workflow.append(name)
                    setattr(
                        self,
                        name,
                        tf.keras.layers.MaxPooling1D(pool_size))

            elif layer.startswith('G'):
                # get the configs
                _, gru_units = layer.split('_')
                gru_units = int(gru_units)

                # set the attributes
                name = 'G' + str(idx)
                self.workflow.append(name)
                setattr(
                    self,
                    name,
                    lambda x: gru(gru_units)[0])

            elif layer.startswith('A'):
                # get the configs
                _, gru_units, attention_units = layer.split('_')
                gru_units = int(gru_units)
                attention_units = attention_units

                # set the attributes
                name = 'A' + str(idx)
                self.workflow.append(name)
                setattr(
                    self,
                    name,
                    GRUAttention(gru_units, attention_units))

            elif layer.startswith('D'):
                # get the configs
                _, units = layer.split('_')
                units = int(units)

                # set the attributes
                name = 'D' + str(idx)
                self.workflow.append(name)
                setattr(
                    self,
                    name,
                    tf.keras.layers.Dense(
                        units,
                        kernel_regularizer=tf.keras.regularizers.l2(0.01),
                        bias_regularizer=tf.keras.regularizers.l2(0.01)))

            elif layer.startswith('O'):
                # get the configs
                _, rate = layer.split('_')
                rate = float(rate)

                # set the attributes
                name = 'O' + str(idx)
                self.workflow.append(name)
                setattr(
                    self,
                    name,
                    tf.layers.Dropout(rate))

            elif layer == 'F':
                # set the attributes
                name = 'F' + str(idx)
                self.workflow.append(name)
                setattr(
                    self,
                    name,
                    tf.layers.flatten)

            elif layer in ['tanh', 'relu', 'sigmoid', 'leaky_relu', 'elu',
                          'softmax']:
                name = 'X' + str(idx)
                self.workflow.append(name)
                if layer == 'tanh':
                    setattr(self, name, tf.tanh)
                elif layer == 'relu':
                    setattr(self, name, tf.nn.relu)
                elif layer == 'sigmoid':
                    setattr(self, name, tf.sigmoid)
                elif layer == 'leaky_relu':
                    setattr(self, name, tf.nn.leaky_relu)
                elif layer == 'elu':
                    setattr(self, name, tf.nn.elu)
                elif layer == 'softmax':
                    setattr(self, name, tf.nn.softmax)

            else:
                raise ValueError(str(layer))

    @tf.function
    def _call(self, x):
        for name in self.workflow:
            x = getattr(self, name)(x)
        return x

    def call(self, x):
        return self._call(x)



    def switch(self, to_test = True):
        if to_test == True:
            for idx, name in enumerate(self.workflow):
                if name.startswith('O'):
                    setattr(self, name, lambda x: x)
        else:
            for idx, value in enumerate(self.fcs):
                if isinstance(value, float):
                    assert value < 1
                    name = 'O' + str(idx)
                    setattr(self, name,
                       tf.layers.Dropout(value))
