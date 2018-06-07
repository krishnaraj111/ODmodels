# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 16:32:03 2018

@author: ckreddy
"""


import xml_to_csv
import generate_tfrecord as gt
import trainck as t


##training data
csv_input='images/train_labels.csv'
image_dir='images/train'
output_path='train.record'
gt.main(csv_input,image_dir,output_path)


##test data
csv_input='images/test_labels.csv'
image_dir='images/test'
output_path='test.record'
gt.main(csv_input,image_dir,output_path)



train_dir='training/'
pipeline_config_path='training/faster_rcnn_inception_v2_pets.config'
task=0
model_config_path=''
train_config_path=''
input_config_path=''
t.main(train_dir,task,pipeline_config_path,model_config_path,train_config_path,input_config_path)



""""
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
"""


