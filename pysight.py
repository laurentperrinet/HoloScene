"""
pysight.py : Testing the python iSight wrapper.


this depends on OpenCV, which intalls fine with MacPorts: {{{
sudo port install opencv +python26 +tbb
}}}

Laurent Perrinet, 2010. Credits: see http://www.incm.cnrs-mrs.fr/LaurentPerrinet/SimpleCellDemo

$Id$

"""
import cv
import time

cv.NamedWindow("camera", 1)

capture = cv.CaptureFromCAM(0)

while True:
    img = cv.QueryFrame(capture)
    cv.ShowImage("camera", img)
    if cv.WaitKey(10) == 27:
        break
