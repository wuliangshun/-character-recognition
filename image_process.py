
# coding: utf-8


"""
Images Process

"""
import tensorflow as tf
from tensorflow.python.ops import sparse_ops
import matplotlib.pyplot as plt
import numpy as np
import os



def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        #dict = pickle.load(fo, encoding='bytes')
        dict = pickle.load(fo)
    return dict


def _int_64_feature(value):
    if isinstance(value, int):
        return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))
    else:
        return tf.train.Feature(int64_list = tf.train.Int64List(value=value))


def _float_feature(value):
    if isinstance(value, float):
        return tf.train.Feature(int64_list = tf.train.FloatList(value=[value]))
    else:
        return tf.train.Feature(int64_list = tf.train.FloatList(value=value))

def _bytes_feature(value):  
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))  

    
def load_cifar_dataset(image_root):
    """
    -----data structure-----
    batch_dict:
        b'labels':               a 10000 int list, range 0-9
        b'batch_label':          b'training batch 1 of 5'
        b'filenames':            a 10000 bytes(string) list
        b'data':                 10000x3072 numpy array , 3072: 32*32*3,  the first 1024 entries contain the red channel, second green, last blue...
    meta_dict:
        b'num_vis':              3072
        b'num_cases_per_batch':  10000
        b'label_names':          [b'airplane', b'automobile', b'bird', b'cat', b'deer', b'dog', b'frog', b'horse', b'ship', b'truck']
    output:
        train_batches:           list of train_batch_dict
        test_batch:              test_batch_dict
        meta:                    meta_dict
    """
    # train batches
    train_batches = []
    for i in range(5):        
        batch_dict = unpickle(image_root + 'data_batch_' + str(i+1))
        train_batches.append(batch_dict)
    # test_batch
    test_batch = unpickle(image_root + 'test_batch')
    # meta
    meta = unpickle(image_root + 'batches.meta')
    
    print('load cifar dataset done !')
    
    return train_batches,test_batch,meta

def write_cifar_to_tfrecords(train_batches,test_batch,tfrecords_root):
    """
    example:
        a sample of one image: 
            'data': 3072 int list
            'label': int
    """
    # 写TFRecord
    
    if not os.path.exists(tfrecords_root):  
        os.makedirs(tfrecords_root)  
        
    # train batches
    writer = tf.python_io.TFRecordWriter(tfrecords_root + 'train_batches.tfrecords')
    for i in range(len(train_batches)):
        batch_dict = train_batches[i]        
        for j in range(len(batch_dict[b'labels'])):
            example = tf.train.Example(features=tf.train.Features(feature={
                        'data':_int_64_feature(batch_dict[b'data'][j]),
                        'label':_int_64_feature(batch_dict[b'labels'][j])
                    }))
            writer.write(example.SerializeToString())
    writer.close()
    
    # test batch
    batch_dict = test_batch
    writer = tf.python_io.TFRecordWriter(tfrecords_root + 'test_batch.tfrecords')
    for j in range(len(batch_dict[b'labels'])):
        example = tf.train.Example(features=tf.train.Features(feature={
                    'data':_int_64_feature(batch_dict[b'data'][j]),
                    'label':_int_64_feature(batch_dict[b'labels'][j])
                }))
        writer.write(example.SerializeToString())
    writer.close()
    
    print('write cifar to TFRecords done !')    
    pass

def read_tfrecords_to_cifar(tfrecords_path, image_num=10000, image_data_length=3072):
    '''
    image_num:          the number of images (default:10000)
    image_data_length:  image data length (default: 32*32*3 = 3072)    
    '''
    images_list = []
    labels_list = []
    reader = tf.TFRecordReader()    
    filename_queue = tf.train.string_input_producer([tfrecords_path])    
    # queue = tf.QueueBase.from_list(0, [train_queue, test_queue])
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
                serialized_example,
                features={
                            'data':tf.FixedLenFeature([image_data_length],tf.int64),
                            'label':tf.FixedLenFeature([],tf.int64)
                        }
            )
    
    datas = tf.cast(features['data'],tf.int64)
    labels = tf.cast(features['label'],tf.int64)
    
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        for i in range(image_num):
            data,label = sess.run([datas,labels])
            images_list.append(data)
            labels_list.append(label)
    print('write cifar to TFRecords done !')    
    
    return images_list,labels_list
    
            
    
def draw_img(img_data):
    """
        image_data: numpy arrays
    """
    plt.imshow(img_data.squeeze())
    plt.show()
        
def write_fonts_to_tfrecords(dir_name,save_dir,save_name,width=224,high=224):
    if not os.path.exists(save_dir):  
        os.makedirs(save_dir)  
    writer = tf.python_io.TFRecordWriter(save_dir+save_name) 
    with tf.Session() as sess:
        for root,dirs,files in os.walk(dir_name):
            for file in files:
                print(file)
                label = file.split('-')[0]
                image_path = root + '/' + file
                #read
                image_data = tf.gfile.FastGFile(image_path,'rb').read()
                #decode: width,height,channels 
                image_data = tf.image.decode_jpeg(image_data)   
                print(image_data.eval().shape)
                #change data type
                image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)  
                #resize
                image_data = tf.image.resize_images(image_data,[width, high], method=0)  
                #to string
                image_raw = sess.run(image_data).tostring()
                
                #TFRecord
                example = tf.train.Example(features=tf.train.Features(feature={  
                    'image_raw': _bytes_feature(image_raw),
                    'label': _int_64_feature(int(label)) 
                }))  
                
                #write TFRecord
                writer.write(example.SerializeToString())  
    writer.close()  
    print('write fonts to TFRecords done !')    

def read_tfrecords_to_fonts(file_name, width=224, high=224):
    
    images_list = []
    labels_list = []
    
    filename_queue = tf.train.string_input_producer([file_name],shuffle=False)
    reader = tf.TFRecordReader()
    _,serialized_example=reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
			'image_raw':tf.FixedLenFeature([],tf.string),
			'label':tf.FixedLenFeature([],tf.int64),
        })
	# decode
    images = tf.decode_raw(features['image_raw'],tf.uint8)
    labels = tf.cast(features['label'],tf.int64)
		
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        
        for i in range(100):
            label,image = sess.run([labels,images])
            # to numpy
            image = np.fromstring(image, dtype=np.float32)
            # reshape
            image = tf.reshape(image,[224,224,3])
            # to uint8
            image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
            ## encode to jpeg
            #image = tf.image.encode_jpeg(image)  
            ## save image
            #with tf.gfile.GFile('./ouput_pic/pic_%d.jpg' % label, 'wb') as f:  
            #f.write(sess.run(image)) 
            images_list.append(image)
            labels_list.append(label)
	
    return images_list,labels_list
  
def main_cifar():
    train_batches,test_batch,meta = load_cifar_dataset(image_root=r'./cifar-10-python/cifar-10-batches-py/')
    write_cifar_to_tfrecords(train_batches,test_batch,tfrecords_root='./Cifar_TFRecords/')
    images_train,labels_train = read_tfrecords_to_cifar(tfrecords_path='./Cifar_TFRecords/train_batches.tfrecords', image_num=10000, image_data_length=3072)

if __name__=='__main__':
    write_fonts_to_tfrecords(dir_name='./final/train/',save_dir='./Font_TFRecords/',save_name='train.tfrecords', width=224, high=224)
    write_fonts_to_tfrecords(dir_name='./final/val/',save_dir='./Font_TFRecords/',save_name='valid.tfrecords', width=224, high=224)
    images_train,labels_train = read_tfrecords_to_fonts(file_name='./Font_TFRecords/train.tfrecords', width=224, high=224)
    pass
    
        
    




