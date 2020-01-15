import tensorflow as tf
strategy = tf.distribute.MirroredStrategy()
print(strategy.num_replicas_in_sync)
