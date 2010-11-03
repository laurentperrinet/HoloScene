#!/usr/bin/env python

"""

SimpleReceptiveField.py : linking the webcam to a crude-but-adaptive neuron and
let it spike in the loudspeaker.

Laurent Perrinet, 2010. Credits: see http://www.incm.cnrs-mrs.fr/LaurentPerrinet/SimpleCellDemo

this depends on OpenCV + numpy + pil, which intalls fine with MacPorts: {{{
sudo port install opencv +python26 +tbb
sudo port install py26-numpy py26-scipy py26-pil py26-distribute py26-pip python_select
sudo python_select python26
sudo port install py26-ipython py26-matplotlib py26-mayavi py26-pyobjc2-cocoa  # optionnally
}}}

Credits: see pysight.py and EgoViewer.py

$Id$

"""
try:
    import pyaudio
    AUDIO = True
except:
    print('Could not import pyaudio, disabling sound')
    AUDIO = False

from numpy import zeros, linspace, hstack, transpose, pi

import cv

import numpy as np
import time

# TODO : plot histogram and membrane potential
# TODO : remove background of RF

#########################################################
#########################################################
NUM_SAMPLES = 1024
# neural parameters
spike = 255*np.ones(45) # that's a crude spike!
quant = 512
rate = 0.01
adaptive = True
sample = 5
sigma = 8.

import numpy as np

#============================================================================
# Set up input
#============================================================================
def cv2array(im):
    depth2dtype = {
        cv.IPL_DEPTH_8U: 'uint8',
        cv.IPL_DEPTH_8S: 'int8',
        cv.IPL_DEPTH_16U: 'uint16',
        cv.IPL_DEPTH_16S: 'int16',
        cv.IPL_DEPTH_32S: 'int32',
        cv.IPL_DEPTH_32F: 'float32',
        cv.IPL_DEPTH_64F: 'float64',
    }

    arrdtype = im.depth
    a = np.fromstring(
         im.tostring(),
         dtype=depth2dtype[im.depth],
         count=im.width*im.height*im.nChannels)
    a.shape = (im.height, im.width, im.nChannels)

    return a

def retina(img, ret=None):
    """
    dummy retina:
     - smooth
     - derive

    """
    dst = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_32F, 3)
#    ret_ = cv.CloneImage(ret)
# format = (frame.width,frame.height)
# frameHSV = cv.CreateImage(format, cv.IPL_DEPTH_8U, 3)
# 
# # calculate HSV and split
# cv.CvtColor(frame, frameHSV, cv.CV_BGR2HSV)
# cv.Split(frameHSV, frameH, frameS, frameBW, None)
# cv.Copy(frame, frameShow)
# 
#     cv.CvtColor(im, im, cv.CV_RGB2GRAY)
    cv.Laplace(img, dst)
    cv.Smooth(dst, dst, smoothtype=cv.CV_GAUSSIAN, param1=7, param2=0, param3=0, param4=0)

#     image = np.flipud(np.fliplr(cv2array(dst)))
#     image /= np.max(np.abs(image))
#     if not(ret == None): 
# #         print cv.GetSize(img), cv.GetSize(ret)
#         cv.AddWeighted(dst, 1, ret, -1., 0., dst)

    return dst # cv.GetMat(dst)


def do_RF(init=False):
#         if not(os.path.isfile('RF.png'))
# 
#         x, y = np.mgrid[-1:1:1j*img.height , -1:1:1j*img.width]
#         mask = np.exp(-.5*(x**2/(.15*4/3)**2 +y**2/.15**2))
#         mat = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_32F, 3)
# # cv.CreateMat(img.height, img.width, cv.CV_32FC1)
#         cv.SetData(mat, cv.fromarray(mask).tostring(), 12*img.width)
         #cv.GetImage()
#         dst = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_32F, 3)

#         cv.Mul(RF, mat, RF, 1.)
# 
    img = cv.QueryFrame(capture)
# TODO:        img = cv.GetMat(cv.QueryFrame(capture))
    if init:
        RF = retina(img, ret)
    else:
        RF = retina(img)
#       mat = cv.fromarray(RF)
#         cv.Resize(RF, img_, interpolation=cv.CV_INTER_LINEAR)
    cv.ShowImage("Receptive Field", RF) # cv.fromarray(RF, allowND=True))

    return RF

#============================================================================
# Create the model.
#============================================================================
corr = 0
voltage = 0.5
hist = np.ones(quant) / quant
result = cv.CreateMat(1, 1, cv.CV_32FC1)

def neuron(im, voltage, hist):
    if voltage > 1.: voltage = 0.
    cv.MatchTemplate(im, RF, result, cv.CV_TM_CCORR_NORMED)
    corr = result[0, 0]
    quantile = int(((corr+1)/2) * quant)-1
    if adaptive:
        cumhist = np.cumsum(hist)
        voltage = cumhist[quantile]
        if voltage > .9:
            if AUDIO: stream.write(spike)
            voltage = 2.

        hist[quantile] += rate
        hist /= np.sum(hist)

    else:
        if corr > .15:
            if AUDIO: stream.write(spike)

    return corr, voltage
#============================================================================
if __name__ == "__main__":
#     print "OpenCV version: %s (%d, %d, %d)" % (cv.VERSION,
#                                                cv.MAJOR_VERSION,
#                                                cv.MINOR_VERSION,
#                                                cv.SUBMINOR_VERSION)
#  
#     print "Press ESC to exit ..."
#  
# 

    if AUDIO:
        # open audio stream to the speakers
        p = pyaudio.PyAudio()
        # initialize loudspeaker
        stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    output=True)

    snapshotTime = time.time()
    capture = cv.CaptureFromCAM(1)
    # check if capture device is OK
    if not capture:
        print "Error opening capture device"
        sys.exit(1)

    downsize = 4

    img_ = cv.QueryFrame(capture)

    cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT, img_.height / downsize)
    cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH, img_.width / downsize)
    cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FORMAT, cv.IPL_DEPTH_32F)

    font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, thickness=2, lineType=cv.CV_AA)
    font_ = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, thickness=3, lineType=cv.CV_AA)


    print ' Startup time ', (time.time() - snapshotTime)*1000, ' ms'
    snapshotTime = time.time()


#     print img.height / 8, img.width / 8 #img.rows / 8, img.cols / 8
#     small = cv.CreateImage((img.height / 8, img.width / 8), cv.IPL_DEPTH_16S, 3)
#     cv.Resize(img, small)
#     dst = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_32F, 3)

    try:
#         ret = cv.CreateImage(cv.GetSize(img_), cv.IPL_DEPTH_32F, 3)
#         cv.Set(ret, 0.)
# 
        cv.NamedWindow("Receptive Field", 0)
        RF = do_RF()
        cv.NamedWindow("Retina", 0)
#         cv.NamedWindow("Info", 1)
#         cv.MoveWindow("Info", 3*RF.width, 0)
        cv.MoveWindow("Receptive Field", 0*RF.width , 0)
        cv.ResizeWindow("Receptive Field", 2*RF.width, 2*RF.height)
#         ret = cv.CreateImage(cv.GetSize(RF), cv.IPL_DEPTH_32F, 3)
        cv.Set(RF, 0.)
        cv.PutText(RF, 'SimpleCellDemo', (12, 48), font, cv.RGB(0, 255, 0))
        cv.PutText(RF, 'Press Esc to exit', (12, 96), font, cv.RGB(255, 0, 0))
        cv.PutText(RF, 'Press r to (re)draw', (12, 144), font, cv.RGB(0, 0, 255))
        cv.ShowImage("Receptive Field", RF)
        img = cv.QueryFrame(capture)
        ret = retina(img)
        cv.ShowImage("Retina", ret)
        cv.ResizeWindow("Retina", 2*RF.width, 2*RF.height)
        cv.MoveWindow("Retina", 2*RF.width, 0)

        while True:
            snapshotTime = time.time()
            img = cv.QueryFrame(capture)
            ret = retina(img, ret)
#             cv.Resize(img, small)
            corr, Vm = neuron(ret, voltage, hist)
            backshotTime = time.time()
            fps = 1. / (backshotTime - snapshotTime)
#             cv.Resize(ret, img_, interpolation=cv.CV_INTER_LINEAR)
            cv.PutText(ret, str('%d'  %fps) + ' fps', (12, 24), font_, cv.RGB(255, 255, 255))
            cv.PutText(ret, str('%d'  %fps) + ' fps', (12, 24), font, cv.RGB(0, 0, 0))

            cv.ShowImage("Retina", ret)
            key = cv.WaitKey(1)
#             if not(key == -1): print key
            if key == 114: do_RF()
            if key == 27: break

    finally:
        # Always close the camera stream
        if AUDIO: stream.close()


