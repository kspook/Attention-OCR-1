import random, time, io, os, shutil, math, sys, logging
import tensorflow as tf

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and return it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def run():
#def main(args):
#    frozen_graph_filename =  os.path.join('trainf1Sv2','attn.pbtxt')
    frozen_graph_filename =  os.path.join('./','frozen_graph.pb')								
    graph=load_graph(frozen_graph_filename)

	
    for op in graph.get_operations(): 
#        print (op.name(), op.value())	
        #if 'Softmax' in op.name:
            #print (op.name)	
        print([s for s in op.name if 'Softmax' in s] )
#if __name__ == "__main__":
#    main(sys.argv[1:])
#    run()
if __name__ == "__main__":
    run()	
	
	
	
'''
You can get all of the node names in your model with something like:

node_names = [node.name for node in tf.get_default_graph.as_graph_def().node]

Or with restoring the graph:

saver = tf.train.import_meta_graph(/path/to/meta/graph)
sess = tf.Session()
saver.restore(sess, /path/to/checkpoints)
graph = sess.graph
print([node.name for node in graph.as_graph_def().node])
'''	