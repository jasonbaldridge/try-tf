import tensorflow.python.platform

import numpy as np
import tensorflow as tf

# Global variables.
NUM_LABELS = 2    # The number of labels.
BATCH_SIZE = 100  # The number of training examples to use per training step.
SEED = None       # Set to None for random seed.

tf.app.flags.DEFINE_string('train', None,
                           'File containing the training data (labels & features).')
tf.app.flags.DEFINE_string('test', None,
                           'File containing the test data (labels & features).')
tf.app.flags.DEFINE_integer('num_epochs', 1,
                            'Number of passes over the training data.')
tf.app.flags.DEFINE_integer('num_hidden', 1,
                            'Number of nodes in the hidden layer.')
tf.app.flags.DEFINE_boolean('verbose', False, 'Produce verbose output.')
FLAGS = tf.app.flags.FLAGS

# Extract numpy representations of the labels and features given rows consisting of:
#   label, feat_0, feat_1, ..., feat_n
def extract_data(filename):

    # Arrays to hold the labels and feature vectors.
    labels = []
    fvecs = []

    # Iterate over the rows, splitting the label from the features. Convert labels
    # to integers and features to floats.
    for line in file(filename):
        row = line.split(",")
        labels.append(int(row[0]))
        fvecs.append([float(x) for x in row[1:]])

    # Convert the array of float arrays into a numpy float matrix.
    fvecs_np = np.matrix(fvecs).astype(np.float32)

    # Convert the array of int labels into a numpy array.
    labels_np = np.array(labels).astype(dtype=np.uint8)

    # Convert the int numpy array into a one-hot matrix.
    labels_onehot = (np.arange(NUM_LABELS) == labels_np[:, None]).astype(np.float32)

    # Return a pair of the feature matrix and the one-hot label matrix.
    return fvecs_np,labels_onehot

def main(argv=None):
    # Be verbose?
    verbose = FLAGS.verbose
    
    # Get the data.
    train_data_filename = FLAGS.train
    test_data_filename = FLAGS.test

    # Extract it into numpy arrays.
    train_data,train_labels = extract_data(train_data_filename)
    test_data, test_labels = extract_data(test_data_filename)

    # Get the shape of the training data.
    train_size,num_features = train_data.shape

    # Get the number of epochs for training.
    num_epochs = FLAGS.num_epochs

    # Get the size of layer one.
    num_hidden = FLAGS.num_hidden
 
    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    x = tf.placeholder("float", shape=[None, num_features])
    y_ = tf.placeholder("float", shape=[None, NUM_LABELS])
    
    # For the test data, hold the entire dataset in one constant node.
    test_data_node = tf.constant(test_data)

    # Define and initialize the network.

    # Initialize the hidden weights and biases.
    w_hidden = tf.Variable(
        tf.truncated_normal([num_features, num_hidden],
                            stddev=0.1,
                            seed=SEED))

    b_hidden = tf.Variable(tf.constant(0.1, shape=[num_hidden]))

    # The hidden layer.
    hidden = tf.nn.relu(tf.matmul(x,w_hidden) + b_hidden)

    # Initialize the output weights and biases.
    w_out = tf.Variable(
        tf.truncated_normal([num_hidden, NUM_LABELS],
                            stddev=0.1,
                            seed=SEED))
    
    b_out = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

    # The output layer.
    y = tf.nn.softmax(tf.matmul(hidden, w_out) + b_out)
    
    # Optimization.
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    
    # Evaluation.
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Create a local session to run this computation.
    with tf.Session() as s:
        # Run all the initializers to prepare the trainable parameters.
    	tf.initialize_all_variables().run()
    	if verbose:
    	    print 'Initialized!'
    	    print
    	    print 'Training.'
    	    
    	# Iterate and train.
    	for step in xrange(num_epochs * train_size // BATCH_SIZE):
    	    if verbose:
    	        print step,
    	        
    	    offset = (step * BATCH_SIZE) % train_size
    	    batch_data = train_data[offset:(offset + BATCH_SIZE), :]
    	    batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
    	    train_step.run(feed_dict={x: batch_data, y_: batch_labels})
    	    if verbose and offset >= train_size-BATCH_SIZE:
    	        print
    	print "Accuracy:", accuracy.eval(feed_dict={x: test_data, y_: test_labels})
            
if __name__ == '__main__':
    tf.app.run()
