# Deze code is gemaakt met de tutorial die te vinden is op https://www.youtube.com/watch?v=2FmcHiLCwTU

# Een aantal zaken zijn depricated, en sommige dingen uit de tutorial bestaan straight-up niet meer, daarvoor
# is het een en ander aangepast. De eerste keer kan traag zijn omdat tf een omweg zoekt voor de depricated stuff.

# Na de run kan je met het commando "tensorboard --logdir=/tmp/logs" tensorboard starten
# Tensorboard kan je bekijken door in je browser naar "localhost:6006" te gaan.

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as inputData
mnist = inputData.read_data_sets("/tmp/data/", one_hot=True)

def main():
    learningRate = 0.01
    iterations = 30
    batchSize = 100
    displayStep = 2

    x = tf.placeholder("float", [None, 784])
    y = tf.placeholder("float", [None, 10])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    with tf.name_scope("Wx_b") as scope:
        model = tf.nn.softmax(tf.matmul(x, W) + b)

    w_h = tf.summary.histogram("weights", W)
    b_h = tf.summary.histogram("biases", b)

    with tf.name_scope("costFunction") as scope:
        costFunction = -tf.reduce_sum(y*tf.log(model))
        tf.summary.scalar("costFunction", costFunction)
    
    with tf.name_scope("train") as scope:
        optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(costFunction)

    init = tf.initialize_all_variables()

    mergedSummary = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(init)
        summaryWriter = tf.summary.FileWriter("/tmp/logs", graph_def = sess.graph_def)

        for iteration in range(iterations):
            avgCost = 0.0
            totalBatch = int(mnist.train.num_examples/batchSize)

            for i in range(totalBatch):
                batchXs, batchYs = mnist.train.next_batch(batchSize)
                sess.run(optimizer, feed_dict={x: batchXs, y: batchYs})
                avgCost += sess.run(costFunction, feed_dict={x: batchXs, y: batchYs})/totalBatch

                summaryStr = sess.run(mergedSummary, feed_dict={x: batchXs, y: batchYs})
                summaryWriter.add_summary(summaryStr, iteration*totalBatch + 1)


            if iteration % displayStep ==0:
                print("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avgCost))

        print("training done")

        predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(predictions,"float"))

        print("Accuracy:", accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))

if __name__ == "__main__":
    main()