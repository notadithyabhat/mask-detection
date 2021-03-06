# mask-detection
Mask detector algorithm to detect the usage of mask and to warn about incorrectly worn masks.

![](demo/demo.gif)

The cases of coronavirus around ther world are constantly on a rise and the need for social distancing and usage of mask is greater than ever. However, only a small populations are taking these mesages and using masks in public. Though there are many mask detectors available online, they don't accurately detect masks which are worn incorrectly(like not covering the nose and so on) which is a mistake that can be notices way too often. This prompted me to develop a mask detector that can detect not only if a mask is worn or not but also detect if the masks are worn correctly.

A custom dataset was collected from friends and family through Google forms for all the 3 cases as there was not enough dataset on masks worn incorrectly online. I'd like to thank each and everyone who contributed to the dataset. Further, some images from datasets from Kaggle were also included in my dataset. There are a total of 826 images in the dataset and augmented images were generated to further enlarge the dataset.

The model was developed using pretrained mobilenetV2 and facenet models with additonal layer with dropout to achieve a validation accuracy of 95.7% validation accuracy. The final model was trained with 99.99% of the dataset and achieved an accuracy of 99%.


Instruction to run:

1. python -m pip install -r requirements.txt


2. CMD:python train_mask_detector.py -d Dataset


3. CMD:python mask-detecctor.py



