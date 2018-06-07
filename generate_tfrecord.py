"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('image_dir', '', 'Path to the image directory')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'PO_NUMBER':
        return 1
    elif row_label == 'PO_DATE':
        return 2
    elif row_label == 'PAYMENT_TERM':
        return 3
    elif row_label == 'CURRENCY_CODE':
        return 4
    elif row_label == 'BILL_TO_ADDRESS':
        return 5
    elif row_label == 'SHIP_TO_ADDRESS':
        return 6
    elif row_label == 'LOGO':
        return 7
    elif row_label == 'SUPPLIER':
        return 8
    elif row_label == 'LINE':
        return 9
    elif row_label == 'LINE_REF':
        return 10
    elif row_label == 'ITEM_DESC':
        return 11
    elif row_label == 'PART_NUMBER':
        return 12
    elif row_label == 'REQUEST_DATE':
        return 13
    elif row_label == 'QTY':
        return 14
    elif row_label == 'UOM':
        return 15
    elif row_label == 'LIST_PRICE':
        return 16
    elif row_label == 'NET_PRICE':
        return 17
    elif row_label == 'ORDER_TOTAL':
        return 18
    elif row_label == 'QUOTE_ID':
        return 19
    elif row_label == 'CONTACT_INFORMATION':
        return 20
    elif row_label == 'DISCOUNT':
        return 21
    elif row_label == 'EFFECTIVE_START_DATE':
        return 22
    elif row_label == 'END_CUSTOMER_ADDRESS':
        return 23
    elif row_label == 'FOB':
        return 24
    elif row_label == 'FREIGHT_CHARGES':
        return 25
    elif row_label == 'EFFECTIVE_END_DATE':
        return 26
    elif row_label == 'PO_TYPE':
        return 27
    elif row_label == 'NOTES':
        return 28
    elif row_label == 'SHIPPING_NOTES':
        return 29
    elif row_label == 'DEAL_ID':
        return 30
    elif row_label == 'LINE_NOTES':
        return 31
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main(csv_input,image_dir,output_path):
#def main(_):
    writer = tf.python_io.TFRecordWriter(output_path)
    path = os.path.join(os.getcwd(), image_dir)
    examples = pd.read_csv(csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
