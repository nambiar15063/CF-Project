
# coding: utf-8

# In[38]:


import numpy as np
import tensorflow as tf
import sys


# In[41]:



#OUTPUT_DIR_TRAIN='C:/Users/Admin/Desktop/deep_learning_data/colaborative_filtering/TFRecords_normal_ratings/tf_records_1M/train'
#OUTPUT_DIR_TEST='C:/Users/Admin/Desktop/deep_learning_data/colaborative_filtering/TFRecords_normal_ratings/tf_records_1M/test'


def _add_to_tfrecord(data_sample,tfrecord_writer):
    
    data_sample=list(data_sample.astype(dtype=np.float32))
#     print (np.shape(data_sample))
    
    example = tf.train.Example(features=tf.train.Features(feature={'movie_ratings': float_feature(data_sample)}))                                          
    tfrecord_writer.write(example.SerializeToString())
    

def _get_output_filename(output_dir, idx, name):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))




# In[46]:


def main():
    ''' Writes the .txt training and testing data into binary TF_Records.'''

    SAMPLES_PER_FILES=100
    print ("started")
    training_set, test_set=_get_dataset("C:\\Users\\nambi\\Desktop\\CF\\Project")

    for data_set, name, dir_ in zip([training_set, test_set], ['train', 'test'], ["C:\\Users\\nambi\\Desktop\\CF\\Project\\train", "C:\\Users\\nambi\\Desktop\\CF\\Project\\test"]):
        
        num_samples=len(data_set)
        print (num_samples)
        i = 0
        fidx = 1

        while i < num_samples:
           
            tf_filename = _get_output_filename(dir_, fidx,  name=name)
            print (tf_filename)
            
            with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
                
                j = 0
                
                while i < num_samples and j < SAMPLES_PER_FILES:
                    
                    sys.stdout.write('\r>> Converting sample %d/%d' % (i+1, num_samples))
                    sys.stdout.flush()
    
                    sample = data_set[i]
                    _add_to_tfrecord(sample, tfrecord_writer)
                    
                    i += 1
                    j += 1
                fidx += 1

    print('\nFinished converting the dataset!')

    
    
    
if __name__ == "__main__":

    #main(output_dir=[OUTPUT_DIR_TRAIN,OUTPUT_DIR_TEST])
    main()
            
    









# In[36]:


import pandas as pd
import numpy as np
import gc
training_set = 2


# In[45]:


def convert(data, num_users, num_movies):
    ''' Making a User-Movie-Matrix'''
    
    new_data=[]
    
    for id_user in range(1, num_users+1):
        
        id_movie=data[:,1][data[:,0]==id_user]
        id_rating=data[:,2][data[:,0]==id_user]
#         print (id_movie)
#         print (id_rating)
        ratings=np.zeros(num_movies, dtype=np.uint32)
        ratings[id_movie-1]=id_rating
        if sum(ratings)==0:
            continue
        new_data.append(ratings)

        del id_movie
        del id_rating
        del ratings
        
    return new_data

def get_dataset_1M(ROOT_DIR):
    ''' For each train.dat and test.dat making a User-Movie-Matrix'''
    
    gc.enable()
    print ("enabled")
    global training_set
    
    training_set=pd.read_csv(ROOT_DIR+'\\train.dat', sep=',', header=None, engine='python', encoding='latin-1')
    training_set=np.array(training_set, dtype=np.uint32)
    
    test_set=pd.read_csv(ROOT_DIR+'\\test.dat', sep=',', header=None, engine='python', encoding='latin-1')
    test_set=np.array(test_set, dtype=np.uint32)
     
    num_users=int(max(max(training_set[:,0]), max(test_set[:,0])))
    num_movies=int(max(max(training_set[:,1]), max(test_set[:,1])))

    print (num_users)
    print (num_movies)
    
    training_set=convert(training_set,num_users, num_movies)
    test_set=convert(test_set,num_users, num_movies)
    print ("converted")
    
    return training_set, test_set
    


def _get_dataset(ROOT_DIR):
    print ("am I even reachng?")
    return get_dataset_1M(ROOT_DIR)

