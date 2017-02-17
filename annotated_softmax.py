import tensorflow.python.platform

import numpy as np
import tensorflow as tf

# Global variables.
NUM_LABELS = 2    # The number of labels.
BATCH_SIZE = 100  # The number of training examples to use per training step.

# Define the flags useable from the command line.
tf.app.flags.DEFINE_string('train', None,
                           'File containing the training data (labels & features).')
tf.app.flags.DEFINE_string('test', None,
                           'File containing the test data (labels & features).')
tf.app.flags.DEFINE_integer('num_epochs', 1,
                            'Number of examples to separate from the training '
                            'data for the validation set.')
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

    # Extract it into numpy matrices.
    train_data,train_labels = extract_data(train_data_filename)
    test_data, test_labels = extract_data(test_data_filename)

    # Get the shape of the training data.
    train_size,num_features = train_data.shape

    # Get the number of epochs for training.
    num_epochs = FLAGS.num_epochs

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    # Add a name to include it in the tensor board visualisation.
    x = tf.placeholder("float", shape=[None, num_features], name="x_input")
    y_ = tf.placeholder("float", shape=[None, NUM_LABELS], name="labels")
    
    # These are the weights that inform how much each feature contributes to
    # the classification.
    W = tf.Variable(tf.zeros([num_features,NUM_LABELS]), name="weights")
    b = tf.Variable(tf.zeros([NUM_LABELS]), name="bias")
    with tf.name_scope("Wx_b") as scope:
        y = tf.nn.softmax(tf.matmul(x,W) + b)


    # Optimization.
    with tf.name_scope("xent") as scope:
        cross_entropy = -tf.reduce_sum(y_*tf.log(y))
        ce_summ = tf.scalar_summary("cross entropy", cross_entropy)
    
    with tf.name_scope("train") as scope:
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # Add summary ops to collect data
    w_hist = tf.histogram_summary("weights", W)
    b_hist = tf.histogram_summary("biases", b)
    y_hist = tf.histogram_summary("y", y)



     # For the test data, hold the entire dataset in one constant node.
    test_data_node = tf.constant(test_data)


    # Evaluation.
    with tf.name_scope("test") as scope:
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        accuracy_summary = tf.scalar_summary("accuracy", accuracy)


    # Create a local session to run this computation.
    with tf.Session() as s:

        # Merge all the summaries and write them out to try_tf_logs
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("try_tf_logs", s.graph_def)

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

            if step % num_epochs == 0:
                feed = {x: test_data, y_: test_labels}
                result = s.run([merged, accuracy], feed_dict=feed)
                summary_str = result[0]
                acc = result[1]
                writer.add_summary(summary_str, step)

        # Give very detailed output.
        if verbose:
            print
            print 'Weight matrix.'
            print s.run(W)
            print
            print 'Bias vector.'
            print s.run(b)
            print
            print "Applying model to first test instance."
            first = test_data[:1]
            print "Point =", first
            print "Wx+b = ", s.run(tf.matmul(first,W)+b)
            print "softmax(Wx+b) = ", s.run(tf.nn.softmax(tf.matmul(first,W)+b))
            print
            
        writer.flush()
        writer.close()
        print "Accuracy:", accuracy.eval(feed_dict={x: test_data, y_: test_labels})
    
if __name__ == '__main__':
    tf.app.run()
