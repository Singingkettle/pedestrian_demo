#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# verifyied by Xiaotong Zhao
# The whole system aims to get the pedestrian bounding box using 2-classes py-faster-rcnn

from __future__ import division

"""
use 2-classes softmax to detect person
the network pre-trained on imagenet, fine-tuned on Caltech
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import cv2, caffe
import os, sys, time
import argparse
import urllib2
import multiprocessing as mp
import os.path
import socket


# os.system('ps aux | grep pedestrian_demo.py | cut -c 10-14 | xargs kill -s 9')
dir_path = os.path.dirname(os.path.realpath(__file__))

# the IP address of client
client_ip = '10.108.190.8'
server_ip = '10.117.18.17'
client_port = 1234
server_port = 8888
camera_ip = '10.108.198.95'

CLASSES = ('__background__', 'person')

NETS = {'VGG16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'ZF': ('ZF',
               'ZF_faster_rcnn_final.caffemodel')}

os.system('rm -r ./img')
os.system('mkdir img')

# deprecated
def image_preprocess(img):
    max_intensity = 255.0
    phi = 1
    theta = 1
    # Increase intensity such that dark pixels become much brighter,
    # bright pixels become slightly bright
    img = (max_intensity/phi) * (img/(max_intensity/theta)) ** 0.5
    # Decrease intensity such that dark pixels become much darker,
    # bright pixels become slightly dark
    #img = (max_intensity/phi) * (img/(max_intensity/theta)) ** 2


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN detection system')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=1, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='detection_net', help='Network to use [VGG16]',
                        choices=NETS.keys(), default='VGG16')
                        #choices=NETS.keys(), default='zf')

    args = parser.parse_args()

    return args


def _init_caffe():
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    #cfg.TEST.SCALES = (768,)
    cfg.TEST.SCALES = (600,)
    # It is set through 375/600 = 480/768, where 375 is pascal image height and 600 is py-faster-rcnn default TEST.SCALES
    # Scale the image may change the detection rate
    # For instance, some small object may be recognized as human when using small scale,
    # but it is recognized correctly when scale is bigger.
    # For a closely person, when using small scale, it may be recognized correctly,
    # but it is recognized falsely when scale is bigger, because the system could only see a part of body
    # It is just like the paper: scale-aware...
    cfg.PIXEL_MEANS = np.array([[[107.35780084, 110.15804144, 106.66253259]]])  # 0~10 sets
    args = parse_args()
    prototxt = '../data/pedestrian_detection/faster_rcnn_test.pt'
    assert os.path.exists(prototxt)
    caffemodel = '../data/pedestrian_detection/VGG16_faster_rcnn_final.caffemodel'
    assert os.path.exists(caffemodel)
    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)
    return net


def image_detection(net, img):
    """Detect object classes in an image using pre-computed object proposals."""
    print 'image_detection', os.getpid()
    scores, boxes = im_detect(net, img)
    # scores.shape is (300, 2)
    # boxes.shape is (300, 8)
    # Visualize detections
    NMS_THRESH = 0.3
    cls_ind = CLASSES.index('person')
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    #vis_detections(img, dets, CONF_THRESH)
    return dets


def video_detection(pre_queue, post_queue):
    print 'video_detection process start'
    print 'video_detection', os.getpid()

    net = _init_caffe()

    while True:
        if not pre_queue.empty():
            timer = Timer()
            timer.tic()
            img = pre_queue.get()
            dets = image_detection(net, img)
            #post_queue.put((img, dets))
            post_queue.put(dets)
            timer.toc()
            #print ('Detection took {:.3f}s for '
            #       '{:d} human proposals').format(timer.total_time, dets.shape[0])


def stream_to_frame(disp_queue, pre_queue, post_queue, stream):
    print 'stream_to_frame process start'
    print 'stream_to_frame', os.getpid()
    bytes = ''
    dets = np.zeros((1, 5))
    frame_num = 0
    frame_iter = 4
    while True:
        bytes += stream.read(1024)
        a = bytes.find('\xff\xd8')
        b = bytes.find('\xff\xd9')

        if a != -1 and b != -1:
            frame_num += 1
            jpg = bytes[a:b + 2]
            bytes = bytes[b + 2:]
            img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            print 'frame number:', frame_num
            # if this frame is key-frame, then send to detection module to detect human
            if (frame_num-1) % frame_iter == 0:
                pre_queue.put(img)
            # if receive the processed frame, then use processed frame to replace original frame
            if not post_queue.empty():
                #img, dets = post_queue.get()
                dets = post_queue.get()
            disp_queue.put((img, dets))


def image_display(disp_queue, send_queue, thresh=0.5):
    """Draw detected bounding boxes."""
    print 'image_display process start'
    print 'image_display', os.getpid()
    img_index = 0
    while True:
        if not disp_queue.empty():
            img, dets = disp_queue.get()
            inds = np.where(dets[:, -1] >= thresh)[0]
            bbox_coordinate = np.zeros((len(inds), 4))

            for i in inds:
                bbox = dets[i, :4]
                score = dets[i, -1]
                bbox_coordinate[i] = bbox
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(img, '%.3f' % score, (bbox[0], int(bbox[1] - 2)), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                            (255, 255, 255), 1)
            current_time = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
            cv2.putText(img, current_time, (10, 15), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, 'Number of people: %d' % len(inds), (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                        (255, 255, 255), 1)
            img_index += 1
            #img_url = dir_path + '/img/%04d.jpg' % img_index
            #cv2.imwrite(img_url, img)
            print 'disp_frame %04d.jpg' % img_index
            print '================================'
            send_queue.put(img)
            #cv2.imshow('Human Detection', img)
            #if cv2.waitKey(1) == 27:
            # if cv2.waitKey(0), the display window will wait infinitely,
            # so the wait time argument must greater than 0
            #    os.system('ps aux | grep pedestrian_demo.py | cut -c 10-14 | xargs kill -s 9')


def send_img_to_client(send_queue):
    print 'send data'
    index = 0
    while True:
        if not send_queue.empty():
            # prepare connection to client
            index += 1
            print 'send_frame %04d.jpg' % index
            print '============================='
            # set up socket
            s = socket.socket()
            s.connect((client_ip, client_port))
            img = send_queue.get()
            img_str = cv2.imencode('.jpg', img)[1].tostring()
            # open the imag_file which would be send to client
            #img_url = send_queue.get()
            #f = open(img_url, 'rb')
            #string_stream = f.read(1024)
            #while(string_stream):
            #   s.send(string_stream)
            #   string_stream = f.read(1024)
            s.send(img_str)
            # end the connection
            s.close()


def shutdown_server():
    s = socket.socket()
    s.bind((server_ip, server_port))
    s.listen(10)
    while True:
        sc, address = s.accept()
        signal = sc.recv(1024)
        if signal == 'stop':
            sc.close()
            s.close()
            os.system('ps aux | grep pedestrian_demo_gui_server.py | cut -c 12-16 | xargs kill -s 9')
        sc.close()
    s.close()
            

def test_net(net):
    # Just for model test
    #img = cv2.imread('/home/david/Applications/py-faster-rcnn/data/demo/004545.jpg')
    img = cv2.imread('/home/david/Workspace/AI/pedestrian_detection/pedestrian_1.jpg')
    print img.shape
    #img = plt.imread('/home/david/Workspace/AI/pedestrian_detection/pedestrian_1.jpg')
    # using different imread function generate different detection result!
    image_detection(net, img)
    print img.shape
    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #plt.show()


if __name__ == '__main__':
    #test_net(net)
    #address = '10.108.197.99'
    kill_port = 'netstat -lnp | grep '+server_ip+':%d' % server_port
    line = os.popen(kill_port)
    pids = line.readlines()
    for i in range(len(pids)):
	tmp = pids[i]
        pid_process = tmp.split()[-1]
        pid_process_name = pid_process.split('/')
        pid = pid_process_name[0]
        kill_pid = 'kill -s 9 '+pid
        os.system(kill_pid)

    address = camera_ip
    print 'Camera IP address is %s.\nStart human detection.' % address
    stream = urllib2.urlopen('http://%s:8083/?action=stream' % address)
    CONF_THRESH = 0.98  # only shows score > CONF_THRESH
    disp_queue = mp.Queue(); pre_queue = mp.Queue(); post_queue = mp.Queue(); send_queue = mp.Queue()
    pread = mp.Process(target=stream_to_frame, args=(disp_queue, pre_queue, post_queue, stream))
    pdet = mp.Process(target=video_detection, args=(pre_queue, post_queue))
    pdisp = mp.Process(target=image_display, args=(disp_queue,send_queue,), kwargs=dict(thresh=CONF_THRESH))
    pserver = mp.Process(target=send_img_to_client, args=(send_queue,))
    pshutdown_server = mp.Process(target=shutdown_server)
    pread.start()
    pdet.start()
    pdisp.start()
    pserver.start()
    pshutdown_server.start()
    pread.join()
    pdet.join()
    pdisp.join()
    pserver.join()
    pshutdown_server.join()
