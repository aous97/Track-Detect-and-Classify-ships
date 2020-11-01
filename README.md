# Track-Detect-and-Classify-ships

Author: Mohamed Aous KHADHRAOUI @SeaOwl France
Last Update: 21/08/2020

**********How To**************

This repo is buit upon yolov5: start by installing requirements in requirements.txt and visit the original yolov5 repo in case a yolov5 problem appears.

- Launch classification and detection on a local video:
	BASE RUN: run in terminal:
		"python tracking_video.py --video "PATH_TO_VIDEO.mp4"
	
  Other options:
	- Start at minute:
	"python tracking_video.py --video "PATH_TO_VIDEO.mp4" --start_at_minute MINUTE
	
	- Change classifier or detector:
	"python tracking_video.py --video "PATH_TO_VIDEO.mp4" --detector (yolov5/yolov3) 		--classif (resnet34/ resnet50/ resnet152)
	
PS: Run "python tracking_video.py --help" for more details on functionalities
- Testing if classification and detection works:
	1- edit Test.py and verify that all lines are uncommented
	2- run in terminal: "python Test.py"



*******CONTENT Of Demo*********

Root:

	- Integration files
	- Classify
	- Detect
	- Track
	- YOLOV5 environment files
	- Test.py: testing detection and classification

************Classify***********

- models: weights of the classification models (with precision 
	on marvel test data)
		- "g_net80%_23.pth" : googlenet weights
		- "r_net81%_23.pth" : resnet34 
		- "Res50_84%.pth" : resnet50
		- "Res152_87%.pth" : resnet152


- test_image: images to test that classification works
- config.py: importing libraries and defining classes names
- produce_classification.py: engine to produce a classification
on an image
- test_classification.py: test classification

**************Detect************

- test_image: images for test
- weights: weights for trained faster-RCNN FPN/resnet50, trained 
yolov5 "best.pt" and pretrained yolov3 on coco dataset
- yolov3: files to load and produce yolov3 detection on an image
- test_detection: test yolov3 on test_images

***************Track************

- IoU_tracker.py: implementation of simple IoU tracking
- sort.py: tracking algorithm using Kalman's filters
- running_classifications.py: manages classification on detected tracks

********Integration files*******

- Test_class.py: testing classification and detection models
- produce_detections.py: detect boats on an image using yolov3
or yolov5
- tracking_video.py: integrates detection, tracking and classification
on a local video and shows new video with choice of trained classifiers
and detectors
PS: execute "python tracking_video.py --help" for more details on how to use

*****yolov5 environment files****

- data: scripts to download coco and voc dataset for training and test
- test.py: test yolov5 detections (output and input in folder "inference")
- train.py: train yolov5 (seaa readme.md or yolov5 github for details)
- tutorial.ipynb tutorial on yolov5
