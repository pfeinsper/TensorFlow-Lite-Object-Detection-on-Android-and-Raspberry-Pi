import tensorflow as tf

pb_file = 'model.pb'
graph_def = tf.compat.v1.GraphDef()

with tf.io.gfile.GFile('RealTimeObjectDetection/Tensorflow/workspace/exported_models/my_model/saved_model/saved_model.pb', 'rb') as f:
    graph_def.ParseFromString(f.read())

# Delete weights
for i in reversed(range(len(graph_def.node))):
    if graph_def.node[i].op == 'Const':
        del graph_def.node[i]

graph_def.library.Clear()

tf.compat.v1.train.write_graph(graph_def, "", 'model.pbtxt', as_text=True)
