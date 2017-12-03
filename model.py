
import math
import numpy as np
import tensorflow as tf
from tensorflow_vgg import vgg16
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit
import process as proc
import log




def get_vgg_codes(batch_size=100,data_method='tfrecords(multi-process/threads)'):
    
    '''
    pickle(multi-process/threads)    or  tfrecords
    '''
    
    with tf.Session() as sess:
        
        #Build vgg network
        vgg = vgg16.Vgg16()
        input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
        
        with tf.name_scope('content_vgg'):
            vgg.build(input_)
            
        if data_method == 'pickle(multi-process/threads)':
            file_names = ['./Font_pickle/data.pickle-'+str(i+1) for i in range(4)]
            images,labels = proc.read_pickle_to_fonts_multifiles(file_names, pool_size=-1, width=224,high=224)
        elif data_method == 'tfrecords(multi-process/threads)':
            file_names = ['./Font_TFRecords/data.tfrecords-'+str(i) for i in range(10)]
            images,labels = proc.read_tfrecords_to_fonts_multifiles(file_names,pool_size=-1, width=20,high=20, is_reshape=True)
        else:
            raise ValueError('No such data import method error !')
        batch,vgg_codes = [],None
        
        for i in range(len(images)):
            #resize
            print(images[i].shape)
            image_data = tf.convert_to_tensor(images[i])        
            print(image_data.shape)
            image_data = tf.image.resize_images(image_data,[224, 224], method=0)             
            batch.append(image_data.eval())
           
            if (i+1) % batch_size == 0 or (i+1) == len(images):
                # batch_size*224*224*1 -> batch_size*224*224*3
                feed_tensor = np.concatenate((batch,batch,batch),axis=3)
                # Get the values from the relu6 layer of the VGG network
                codes_batch = sess.run(vgg.relu6, feed_dict={input_:feed_tensor})
                # Building an array of the codes
                if vgg_codes is None:
                    vgg_codes = codes_batch
                else:
                    vgg_codes = np.concatenate((vgg_codes, codes_batch))
                # Reset to start building the next batch
                batch = []
                print('{} images processed'.format(i+1))
                
    #one-hot encoder
    labels = labels[0:vgg_codes.shape[0]]
    lb = LabelBinarizer()
    lb.fit(labels)    
    labels_vecs = lb.transform(labels)
    
    # Shuffle and split dataset    
    ss_train = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
       
    train_idx, test_idx = next(ss_train.split(vgg_codes, labels))

    train_x, train_y = vgg_codes[train_idx], labels_vecs[train_idx]
    test_x, test_y = vgg_codes[test_idx], labels_vecs[test_idx]

    ss_test = StratifiedShuffleSplit(n_splits=10, test_size=0.5)
    val_idx, test_idx = next(ss_test.split(test_x, test_y))

    val_x, val_y = test_x[val_idx], test_y[val_idx]
    test_x, test_y = test_x[test_idx], test_y[test_idx]


    # Save data
    proc.save_pickle(save_dir='./Font_pickle/',save_name='train_x.p',obj=train_x)
    proc.save_pickle(save_dir='./Font_pickle/',save_name='train_y.p',obj=train_y)
    proc.save_pickle(save_dir='./Font_pickle/',save_name='val_x.p',obj=val_x)
    proc.save_pickle(save_dir='./Font_pickle/',save_name='val_y.p',obj=val_y)
    proc.save_pickle(save_dir='./Font_pickle/',save_name='test_x.p',obj=test_x)
    proc.save_pickle(save_dir='./Font_pickle/',save_name='test_y.p',obj=test_y)

def get_batches(x, y, n_batches):
    """ Return a generator that yields batches from arrays x and y. """
    batch_size = len(x)//n_batches
    print(batch_size)
    print(n_batches)

    for ii in range(0, n_batches*batch_size, batch_size):

        # If not on the last batch, grab data with size batch_size
        if ii != (n_batches-1)*batch_size:
            X, Y = x[ii: ii+batch_size], y[ii: ii+batch_size]

        # On the last batch, grab the rest of the data
        else:
            X, Y = x[ii:], y[ii:]
        yield X, Y 

def model(epochs=10,batch_size=20,learning_rate=0.0002,unit_num=2048,keep_prob=0.5,iteration_print=10):
    print('Loading codes and labels...')
    #load data
    train_x = proc.load_pickle('./Font_pickle/train_x.p')
    train_y = proc.load_pickle('./Font_pickle/train_y.p')
    val_x = proc.load_pickle('./Font_pickle/val_x.p')
    val_y = proc.load_pickle('./Font_pickle/val_y.p')
    
    
    # Inputs
    inputs_ = tf.placeholder(tf.float32, shape=[None, train_x.shape[1]], name='inputs')
    labels_ = tf.placeholder(tf.int64, shape=[None, train_y.shape[1]], name='labels')
    
    # Classifier
    fc1 = tf.contrib.layers.fully_connected(inputs_,
                                            unit_num,
                                            weights_initializer=tf.truncated_normal_initializer(stddev=2 / math.sqrt(unit_num)),
                                            #  weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                            biases_initializer=tf.zeros_initializer())
    fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)

    fc2 = tf.contrib.layers.fully_connected(fc1,
                                            int(unit_num / 2),
                                            weights_initializer=tf.truncated_normal_initializer(stddev=2 / math.sqrt(unit_num / 2)),
                                            #  weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                            biases_initializer=tf.zeros_initializer())
    fc2 = tf.nn.dropout(fc2, keep_prob=keep_prob)

    fc3 = tf.contrib.layers.fully_connected(fc2,
                                            int(unit_num / 4),
                                            weights_initializer=tf.truncated_normal_initializer(stddev=2 / math.sqrt(unit_num / 4)),
                                            #  weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                            biases_initializer=tf.zeros_initializer())
    fc3 = tf.nn.dropout(fc3, keep_prob=keep_prob)

    logits = tf.contrib.layers.fully_connected(fc3,
                                               train_y.shape[1],
                                               activation_fn=None)
    with tf.name_scope('cost'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_, logits=logits),name='cost')
    tf.summary.scalar('cost', cost)
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    # Operations for validation/test accuracy
    predicted = tf.nn.softmax(logits,name='predicted')
    correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(labels_, 1))

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
        tf.summary.scalar('accuracy', accuracy)
        
    saver = tf.train.Saver()
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        
        train_writer = tf.summary.FileWriter('./log/train', sess.graph)
        val_writer = tf.summary.FileWriter('./log/val')

        sess.run(tf.global_variables_initializer())
        iteration = 1

        for i in range(epochs):
            for batch_x, batch_y in get_batches(train_x, train_y, batch_size):
                summary_train, loss, train_acc, _ = sess.run([merged, cost, accuracy, optimizer],
                                                             feed_dict={inputs_: batch_x,
                                                             labels_: batch_y})

                train_writer.add_summary(summary_train, iteration)

                if iteration % iteration_print == 0:
                    summary_val, val_acc = sess.run([merged, accuracy], feed_dict={inputs_: val_x,
                                                                                   labels_: val_y})

                    val_writer.add_summary(summary_val, iteration)

                    print('Epochs: {:>3} | Iteration: {:>5} | Loss: {:>9.4f} | Train_acc: {:>6.2f}% | Val_acc: {:.2f}%'
                          .format(i+1, iteration, loss, train_acc * 100, val_acc * 100))

                iteration += 1

        saver.save(sess, "checkpoints/fonts.ckpt")
    
 
        
def test():
    load_path = './checkpoints/fonts.ckpt'
    # Load test codes and labels
    print('Loading test codes and labels...')
    test_x = proc.load_pickle('./Font_pickle/test_x.p')
    test_y = proc.load_pickle('./Font_pickle/test_y.p')
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:

        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)

        inputs_ = loaded_graph.get_tensor_by_name('inputs:0')
        labels_ = loaded_graph.get_tensor_by_name('labels:0')
        accuracy = loaded_graph.get_tensor_by_name('accuracy/accuracy:0')

        feed = {inputs_: test_x,
                labels_: test_y}

        test_acc = sess.run(accuracy, feed_dict=feed)
        log.log_and_print("Test accuracy: {:.4f}".format(test_acc * 100))


        
#        predicted = loaded_graph.get_tensor_by_name('predicted:0')
#        
#        feed = {inputs_: test_x,
#                labels_: test_y}
#        
#        correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(labels_, 1))
#        
#        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
#
#        test_acc = sess.run(accuracy, feed_dict=feed)
#        
#        log.log_and_print("Test accuracy: {:.4f}".format(test_acc * 100))
 
    
if __name__=='__main__':
    get_vgg_codes(batch_size=20,data_method='tfrecords(multi-process/threads)')
    model(epochs=20,batch_size=20,learning_rate=0.0002,unit_num=2048,keep_prob=0.5,iteration_print=10)
    test()
    pass
   