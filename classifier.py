#!/usr/bin/env python

import device_patches # Device specific patches for Jetson Nano (needs to be before importing cv2)

import cv2
import os
import time
import sys, getopt
import numpy as np
from edge_impulse_linux.image import ImageImpulseRunner

model_path = "model/modelfile.eim"
data_path = "data"

runner = None

dir_path = os.path.dirname(os.path.realpath(__file__))
modelfile = os.path.join(dir_path, model_path)

totaltime = 0
files = 0
dim = 96


with ImageImpulseRunner(modelfile) as runner:
    try:
        model_info = runner.init()
        print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
        print("")
        
        labels = model_info['model_parameters']['labels']

        for sample in os.listdir(data_path):
            file = data_path + "/" + sample
            files += 1

            img = cv2.cvtColor(
                cv2.imread(file), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (dim, dim))
            print("Loaded test image: "+file)
            
            features, cropped = runner.get_features_from_image(img)

            start = time.time()
            res = runner.classify(features)
            end = time.time()
            benchmark = end - start
            totaltime += benchmark
            
            ground = "Unknown"
            classifications = res['result']['classification']
            result = max(classifications, key=classifications.get)
            
            if "Benign" in sample:
                ground = "Benign"
            elif "Pre" in sample:
                ground = "Pre"
            elif "Pro" in sample:
                ground = "Pro"
                
            for label in labels:
                score = classifications[label]
                print('%s: %.2f\t' % (label, score), end='')
            print("")
                
            print("Ground: "+ground)
            print("Classification: " + result + " with " + str(classifications[result]) + " confidence")
            
            if ground == result:
                print("Result: Correctly classified " + result + " sample in "  + str(benchmark) + " seconds")
            else:
                print("Result: Incorrectly classified " + result + " sample in "  + str(benchmark) + " seconds")
            print("")
        print("Classifications finished " + str(files) + " in " + str(totaltime) + " seconds")

    finally:
        if (runner):
            runner.stop()