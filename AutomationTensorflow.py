print('importing python packages....')
import numpy as np
import os
import os.path
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
# from matplotlib import pyplot as plt
from PIL import Image,ImageEnhance
import pytesseract
import unittest,time,os
from appium import webdriver
from time import sleep
from autocorrect import spell
import warnings

filename=sys.argv[1]
if os.path.isfile(filename):
    print(filename,'exists -test case will be loaded from it')
    fname=filename
else:
    print(filename, 'does not exist using default file testcase.txt')
    fname="testcase.txt"

warnings.filterwarnings("ignore")

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops


if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

# Environment Setup
from utils import label_map_util
from utils import visualization_utils as vis_util

# Model Set Up
MODEL_NAME = 'textbox1'      # What model to download.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'   # Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt') # List of the strings that is used to add correct label for each box.
NUM_CLASSES = 1
#map of classes
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
PATH_TO_IMAGES_DIR = 'test_images'
IMAGE_PATH = [ os.path.join(PATH_TO_IMAGES_DIR, 'image.jpg')]

desired_caps = {}
desired_caps['platformName'] = 'Android'
# desired_caps['platformVersion'] = '4.4'
desired_caps['deviceName'] = 'BMPZWCPN99999999'
desired_caps['appPackage'] = 'com.db.awm.bauhaus.ebanking'
desired_caps['appActivity'] = 'com.db.awm.bauhaus.ebanking.MainActivity'


# Screenshot related parameters
dir = 'test_images\image7.png'
screenshot_dir=os.path.join(os.getcwd(),dir)

print('Starting tensorflow wait ....')

# Detection Graph
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


#Image to numpy array
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def control_box_text(img,detection_box):
    width,height=img.size
    control_dims={}
    for i in detection_box:
        if i[0] !=0:
            yL=int(i[0]*height)
            xL=int(i[1]*width)
            yR=int(i[2]*height)
            xR=int(i[3]*width)
            control=img.crop([xL,yL,xR,yR])
            #contrast=ImageEnhance.Contrast(control)
            #img=contrast.enhance(3)
            #control.show()
            coord = [xL, yL, xR, yR]
            controlname=(spell(pytesseract.image_to_string(control))).lower()
            control_dims[controlname]=coord
            print()
    return control_dims

def controlLocation(image_path):
    #def controlLocation(image_path):
  image = Image.open(image_path)
  image1= image.convert('RGB') # fix for PNG images
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
#  print('Printing Image Shape',image.shape)
  image_np = load_image_into_numpy_array(image1)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)
  control_dims=control_box_text(image,output_dict['detection_boxes'])
  # Visualization of the results of a detection.
  return control_dims

def click(coord):
    x = (coord[0] + coord[2]) / 2
    y = (coord[1] + coord[3]) / 2
    str = 'adb shell input tap {h} {v}'.format(h=x, v=y)
    os.system(str)

def enter(coord,text):
    click(coord)
    str1 = "adb shell input text '{}'".format(text)
    os.system(str1)
#stared Aoo

def checkScreen():
    print('Grabbing the screen ...')
    driver.save_screenshot(screenshot_dir)
    print('checking the screen with object detector....')
    control_dims = controlLocation(dir)
    return control_dims

teststep=[]

with open(fname) as f:
    content =f.readlines()
    for i in content:
        command=''
        text=''
        control=''
        words=i.split(" ")
        command=(words[0].lower()).strip()
        if command=='enter':
            x=1
            while words[x] not in 'in':
                if not text:
                    text=(words[x]).strip()
                else:
                    text = text +" "+ (words[x]).strip()
                x=x+1
            str=''
            for i in range(x+1,len(words)):
                if not str:
                    str=str+(words[i]).strip()
                else:
                    str=str+" "+(words[i]).strip()
            control=(str.strip()).lower()
            teststep.append([command,text,control])
        elif command =='click':
            str=''
            for i in range(1,len(words)):
                if not str:
                    str=str+(words[i]).strip()
                else:
                    str=str+" "+(words[i]).strip()

            control=(str.strip()).lower()
            teststep.append([command, text, control])
        else:
            pass
    print(teststep)
    f.close()


driver = webdriver.Remote('http://127.0.0.1:4723/wd/hub', desired_caps)
print('waiting for screen .....')
control_dims=checkScreen()

controlList=[]
for i in teststep:
    controlList.append(i[2])
check=controlList[0]
while not (check in control_dims):
    print('Screen not yet available ... will check in a few seconds')
    sleep(5)
    control_dims = checkScreen()

for i in teststep:
    if i[0] in 'enter':
        coord=control_dims[i[2]]
        #print('entering data ', i[2], ' at ', coord)
        enter(coord,i[1])
    elif i[0] in 'click':
        coord = control_dims[i[2]]
        #print('clicking ', i[2],' at ',coord)
        click(coord)




