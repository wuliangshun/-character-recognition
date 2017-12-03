
# coding: utf-8


"""
Images Process

"""
import time
import tensorflow as tf
#from tensorflow.python.ops import sparse_ops
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import log


def save_pickle(save_dir,save_name,obj):
    if not os.path.exists(save_dir):  
        os.makedirs(save_dir)  
    with open(os.path.join(save_dir,save_name),'wb') as f:
        pickle.dump(obj,f)
        
def load_pickle(file):
    with open(file, 'rb') as f:
        #obj = pickle.load(fo, encoding='bytes')
        obj = pickle.load(f)
    return obj


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
        batch_dict = load_pickle(image_root + 'data_batch_' + str(i+1))
        train_batches.append(batch_dict)
    # test_batch
    test_batch = load_pickle(image_root + 'test_batch')
    # meta
    meta = load_pickle(image_root + 'batches.meta')
    
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
     
def write_fonts_to_pickle(dir_name,save_dir,save_name,width=224,high=224,images_scope='all',is_resize=False):
    '''
    images_scope:'all' or ['1','45','32']
    ''' 
    img_data_list = []
    label_list = []
    with tf.Session() as sess:
        for root,dirs,files in os.walk(dir_name):
            for file in files:
                if '.jpg' in file:                    
                    if images_scope == 'all' or file.split('-')[0] in images_scope:    
                        #print(file)
                        label = int(file.split('-')[0])
                        image_path = root + '/' + file
                        #read
                        image_data = tf.gfile.FastGFile(image_path,'rb').read()
                        #decode: width,height,channels 
                        image_data = tf.image.decode_jpeg(image_data)   
                        #change data type
                        image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)  
                        #resize
                        if is_resize:
                            image_data = tf.image.resize_images(image_data,[width, high], method=0)  
                        #save
                        img_data_list.append(image_data.eval())
                        label_list.append(label)
                    
    train = [img_data_list,label_list]
    save_pickle(save_dir,save_name,train)
    print('write fonts to pickle done !')    
    
        
def read_pickle_to_fonts(file_name,width=224,high=224):
    data = load_pickle(file_name)
    [images_list,labels_list] = data 
    return images_list,labels_list
 
def write_fonts_to_tfrecords(dir_name,save_dir,save_name,width=224,high=224,images_scope='all',is_resize=False):

    start_time = time.time()
    
    if not os.path.exists(save_dir):  
        os.makedirs(save_dir)  
        
    writer = tf.python_io.TFRecordWriter(save_dir+save_name) 
    with tf.Session() as sess:
        for root,dirs,files in os.walk(dir_name):
            for file in files:
                if '.jpg' in file:                    
                    if images_scope == 'all' or file.split('-')[0] in images_scope:   
                        print('write tfrecords:{}'.format(file))
                        label = file.split('-')[0]
                        image_path = root + '/' + file
                        #read
                        image_data = tf.gfile.FastGFile(image_path,'rb').read()
                        #decode: width,height,channels 
                        image_data = tf.image.decode_jpeg(image_data)   
                        #change data type
                        image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)  
                        #resize
                        if is_resize:
                            image_data = tf.image.resize_images(image_data,[width, high], method=0)  # image_data = tf.image.resize_image_with_crop_or_pad(img_data,width,high) #this method will padding with 0
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
    
    log.log_and_print('Time: {}s'.format(time.time() - start_time))
    print('write fonts to TFRecords done !')    

def read_tfrecords_to_fonts(file_name, width=224, high=224, is_reshape=False):
    
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
            if is_reshape:
                image = tf.reshape(image,[width,high,1])
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

def read_fonts(dir_name,width=224,high=224, is_resize=False, is_reshape=False):

    start_time = time.time()

    images_list = []
    labels_list = []
    with tf.Session() as sess:
        for root,dirs,files in os.walk(dir_name):
            for file in files:
                if '.jpg' in file:
                    print('read file:{}'.format(file))
                    label = file.split('-')[0]
                    image_path = root + '/' + file
                    #read
                    image_data = tf.gfile.FastGFile(image_path,'rb').read()
                    #decode: width,height,channels 
                    image_data = tf.image.decode_jpeg(image_data)   
                    #change data type
                    image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)  
                    #resize
                    if is_resize:
                        image_data = tf.image.resize_images(image_data,[width, high], method=0) 
                    #to numpy
                    image = image_data.eval()
                    #reshape
                    if is_reshape:
                        image = np.reshape(image,[width,high,1])
                    print(image.shape)
                    #append
                    images_list.append(image)                   
                    labels_list.append(label)
                    
    log.log_and_print('Time: {}s'.format(time.time() - start_time))
    print('read fonts done !')    
    
    return images_list,labels_list
    

def write_fonts_to_pickle_multithreads(pool_size=-1, label_len=[0,2376], dir_name='./Font',save_dir='./Font_pickle/',save_name='data_train.pickle', width=224, high=224, is_resize=False):
    '''
    multi threads to write to pickle
    '''
    import threadpool
    
    if pool_size == -1:
        import multiprocessing
        pool_size = multiprocessing.cpu_count()*2
        
    pool = threadpool.ThreadPool(pool_size)
    
    img_lbls = [str(i) for i in list(range(label_len[0],label_len[1]))]
    span = int(len(img_lbls)/pool_size)
    
    args = []
    for i in range(pool_size):
        start = i*span
        end = (i+1)*span if (i+1)*span > len(img_lbls)-1 else len(img_lbls)-1
        args.append(([dir_name, save_dir, '_'+str(i)+save_name, width, high, img_lbls[start:end], is_resize], None))
    
    reqs = threadpool.makeRequests(write_fonts_to_pickle, args)
    [pool.putRequest(req) for req in reqs]
    pool.wait()
    
    
def write_fonts_to_pickle_multiprocessing(pool_size=-1, label_len=[0,2376], dir_name='./Font',save_dir='./Font_pickle/',save_name='data.pickle', width=224, high=224, is_resize=False):
    '''
    multi process to write to pickle
    '''
    import multiprocessing
    
    if pool_size == -1:
        pool_size = multiprocessing.cpu_count()
    
    img_lbls = [str(i) for i in list(range(label_len[0],label_len[1]))]
    span = int(len(img_lbls)/pool_size)
    
    pool = multiprocessing.Pool(processes=pool_size)
    
    args = []
    for i in range(pool_size):
        start = i*span
        end = (i+1)*span if (i+1)*span > len(img_lbls)-1 else len(img_lbls)-1
        args.append(([dir_name, save_dir, save_name+'-'+str(i), width, high, img_lbls[start:end]], None))
        pool.apply_async(write_fonts_to_pickle, (dir_name, save_dir, save_name+'-'+str(i), width, high, img_lbls[start:end], is_resize,))
        
    pool.close()
    pool.join()

def write_fonts_to_tfrecords_multiprocessing(pool_size=-1, label_len=[0,2376], dir_name='./Font',save_dir='./Font_TFRecords/',save_name='data.tfrecords', width=224, high=224, is_resize=False):
    '''
    multi process to write to tfrecords
    '''
    import multiprocessing
    
    if pool_size == -1:
        pool_size = multiprocessing.cpu_count()
    
    img_lbls = [str(i) for i in list(range(label_len[0],label_len[1]))]
    span = int(len(img_lbls)/pool_size)
    
    pool = multiprocessing.Pool(processes=pool_size)
    
    args = []
    for i in range(pool_size):
        start = i*span
        end = (i+1)*span if (i+1)*span > len(img_lbls)-1 else len(img_lbls)-1
        args.append(([dir_name, save_dir, save_name+'-'+str(i), width, high, img_lbls[start:end]], None))
        pool.apply_async(write_fonts_to_tfrecords, (dir_name, save_dir, save_name+'-'+str(i), width, high, img_lbls[start:end], is_resize,))
        
    pool.close()
    pool.join()

def read_pickle_to_fonts_multifiles(file_names, pool_size=-1, width=224,high=224):    
    
    import multiprocessing
    
    if pool_size == -1:
        pool_size = multiprocessing.cpu_count()
        
    pool = multiprocessing.Pool(processes=pool_size)
    
    result = []
    for file_name in file_names:
        result.append(pool.apply_async(read_pickle_to_fonts, (file_name, width, high, )))
    
    pool.close()
    pool.join()
    
    images_list,labels_list = [],[]
    for res in result:         
         images_temp,labels_temp = res.get(0)
         images_list.extend(images_temp)
         labels_list.extend(labels_temp)
    
    return images_list,labels_list

def read_tfrecords_to_fonts_multifiles(file_names, pool_size=-1, width=224,high=224, is_reshape=False):
    
    from multiprocessing.pool import ThreadPool

    if pool_size == -1:
        import multiprocessing
        pool_size = multiprocessing.cpu_count()
    
    pool = ThreadPool(pool_size)
    
    results = []
    
    for file_name in file_names:
        results.append(pool.apply_async(read_tfrecords_to_fonts, (file_name, width, high, is_reshape))) # tuple of args for foo)
    
    results = [r.get() for r in results]    
    
    pool.close()
    pool.join()
    
    images_list,labels_list = [],[]
    for r in results:        
         images_temp,labels_temp = r
         images_list.extend(images_temp)
         labels_list.extend(labels_temp)
    
    print('read TFRecords multi files done !')
    
    return images_list,labels_list

   
def main_cifar():
    train_batches,test_batch,meta = load_cifar_dataset(image_root=r'./cifar-10-python/cifar-10-batches-py/')
    write_cifar_to_tfrecords(train_batches,test_batch,tfrecords_root='./Cifar_TFRecords/')
    images_train,labels_train = read_tfrecords_to_cifar(tfrecords_path='./Cifar_TFRecords/train_batches.tfrecords', image_num=10000, image_data_length=3072)
    
def main_font():
    
    #pickle
    write_fonts_to_pickle(dir_name='./Font/train/',save_dir='./Font_pickle/',save_name='data_train.pickle', width=224, high=224, is_resize=False)
    write_fonts_to_pickle(dir_name='./Font/val/',save_dir='./Font_pickle/',save_name='data_valid.pickle', width=224, high=224, is_resize=False)
    images_train,labels_train = read_pickle_to_fonts(file_name='./Font_pickle/data_train.pickle', width=224, high=224)
    images_valid,labels_valid = read_pickle_to_fonts(file_name='./Font_pickle/data_valid.pickle', width=224, high=224)  
    
    #multithreads write pickle
    write_fonts_to_pickle_multithreads(pool_size=-1, label_len=[0,2376], dir_name='./Font',save_dir='./Font_pickle/',save_name='data.pickle', width=224, high=224, is_resize=False)    
    
    #multiprocessing write pickle
    write_fonts_to_pickle_multiprocessing(pool_size=-1, label_len=[0,2376], dir_name='./Font',save_dir='./Font_pickle/',save_name='data.pickle', width=224, high=224, is_resize=False)
    
    #multi files read pickle
    images_list,labels_list = read_pickle_to_fonts_multifiles(['./Font_pickle/data.pickle-1','./Font_pickle/data.pickle-2','./Font_pickle/data.pickle-3','./Font_pickle/data.pickle-4'],pool_size=-1, width=224,high=224)
    
    #tfrecords
    write_fonts_to_tfrecords(dir_name='./Font/train/',save_dir='./Font_TFRecords/',save_name='train.tfrecords', width=224, high=224, is_resize=False)
    write_fonts_to_tfrecords(dir_name='./Font/val/',save_dir='./Font_TFRecords/',save_name='valid.tfrecords', width=224, high=224, is_resize=False)
    images_train,labels_train = read_tfrecords_to_fonts(file_name='./Font_TFRecords/train.tfrecords', width=224, high=224, is_reshape=False)
    images_valid,labels_valid = read_tfrecords_to_fonts(file_name='./Font_TFRecords/valid.tfrecords', width=224, high=224, is_reshape=False)
    
    #multiprocessing write tfrecords
    write_fonts_to_tfrecords_multiprocessing(pool_size=10, label_len=[0,2376], dir_name='./Font',save_dir='./Font_TFRecords/',save_name='data.tfrecords', width=224, high=224, is_resize=False)
    
    #direct
    images_list,labels_list = read_fonts(dir_name='./Font')
    
    
if __name__=='__main__':    
    #write_fonts_to_tfrecords_multiprocessing(pool_size=10, label_len=[0,2376], dir_name='./Font',save_dir='./Font_TFRecords/',save_name='data.tfrecords', width=224, high=224, is_resize=False)
    pass
    
        
    




