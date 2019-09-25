# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

# import by nakatani part
# socket connection
import socket
# for get time
from datetime import datetime
# for csv file
import pandas as pd


# 最小限のGPUメモリのみ確保
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "2,3"
sess = tf.Session(config=config)
K.set_session(sess)

class YOLO(object):
    _defaults = {
        "model_path": 'yolov3.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
#        "classes_path": 'model_data/my_classes.txt',
        "score" : 0.80,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 0,
        # 0 for nakatani pc maybe use 2 for gpu
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image, csv_list):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        if len(out_scores) == 0:
            return None, None
        i = 0
        c = out_classes[0]
        print(out_scores)
        # print(i)
        predicted_class = self.class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))
        
        # add csv data written by nakatani
        csv_list.append(label.split()[1])


        # socket block written by nakatani
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # サーバを指定
            s.connect(('127.0.0.1', 90))
            # サーバにメッセージを送る
            # s.sendall(b'hello')
            send_message = label + ":"
            s.sendall(send_message.encode("UTF-8"))


        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])
            
        # My kingdom for a good redistributable image drawing library.
        image = image.crop((left, top, right, bottom))

# show bounding box 171 - 201 written by nakatani        
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            # print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start, "seconds takes")
        return image, csv_list

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        # defined by nakatani
        csv_list = []

        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        detected_image, csv_list = yolo.detect_image(image, csv_list)
        result = np.asarray(detected_image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        
        # バウンディングボックスが0だと強制終了する点の修正 written by nakatani
        if result.all() == None:
            pass
        else:
        # save image written by nakatani
            # get current time
            current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            # define file name
            current_path = os.getcwd()
            file_name =  "yolo_outputs/" + current_time + ".png"
            # save image to png file
            image.save(file_name)
            # add csv list
            csv_list.append(current_time)
            csv_list.append(file_name)
            print(csv_list)
            # save csv file
            df = pd.DataFrame([csv_list],
              columns=['time', 'pred', "file_path"])
            file_list = os.listdir("yolo_outputs")
            if "log.csv" in file_list:
                df.to_csv("yolo_outputs/log.csv", index=False, encoding="utf-8", mode='a', header=False)
            else:
                df.to_csv("yolo_outputs/log.csv", index=False, encoding="utf-8", header=False)
    

            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
            if isOutput:
                out.write(result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

def detect_img(yolo):
    import cv2
    import glob
    #annotation_path = "../【写真データ】ユーザー推定AI開発の為のデータ取得作業_resized_256x256/annotation/annotation2.txt"
    #output_path = "./ano_data/"
    image_path = "./nakatani_image/"
#     image_path = "./test/*/"
    output_path = "./nakatani_output/"
    print("Inputs image " + image_path+"*.png")
    
    i = 0
    
    for img_name in glob.iglob(image_path+"*.png", recursive=True):
        i += 1
        print("Input image " + img_name)
        image = Image.open(img_name)
        
        # written by nakatani
        csv_list = []
        r_image, csv_list = yolo.detect_image(image, csv_list)

        # r_image = yolo.detect_image(image)
        if r_image is None:
            pass
        else:
            # print(type(r_image))
            os.makedirs(output_path + img_name.split('/')[-2] + "/", exist_ok=True)
            print("Outputs image path " + output_path + img_name.split('/')[-2] + "/" + img_name.split('/')[-1])
            r_image.save(output_path + img_name.split('/')[-2] + "/" + img_name.split('/')[-1])
            
#         if i == 10:
#             break
            
        # ret = cv2.imwrite(output_path+img_name.split('/')[-1], np.asarray(r_image)[..., ::-1])
        #if not ret:
        #    print('Failed to write image.')

#     with open(annotation_path) as f:
#         for line in f:
#             print(line.split(' ')[0])
#             img_name = line.split(' ')[0]
#             image = Image.open(img_name)
#             r_image = yolo.detect_image(image)
#             print(type(r_image))
#             print(output_path+img_name.split('/')[-1])
#             ret = cv2.imwrite(output_path+img_name.split('/')[-1], np.asarray(r_image)[..., ::-1])
#             if not ret:
#                 print('Failed to write image.')
#     while True:
#         img = input('Input image filename:')
#         try:
#             image = Image.open(img)
#         except:
#             print('Open Error! Try again!')
#             continue
#         else:
#             r_image = yolo.detect_image(image)
#             print(type(r_image))
#             import cv2
#             cv2.imwrite("out.jpg", np.asarray(r_image)[..., ::-1])
#             r_image.show()
    yolo.close_session()
    
if __name__ == '__main__':
    detect_img(YOLO())
    # detect_video(YOLO(), 1)
