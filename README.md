# Monocyte Monolayer Index calculation using MaskRCNN

Image segmentation has been used in the medical field for long time ago. Artificial intelligence algorithms have an incredible ability to learn and improve themselves. The Monocyte Monolayer Assay (MMA) is an in-vitro procedure that has been used for more than 40 years to predict in-vivo survivability of incompatible RBCs [1]. The MMA mimics extravascular hemolysis of red blood cells and may be used to guide transfusion recommendations. The Monocyte Index is a percentage of RBCs adhered, ingested or both by the monocytes. A minimum of 400 myocytes must be counted for enough data to determine the risk of transfusion of incompatible blood. In this work, Artificial Intelligence has been used to automate the Monocyte Index calculation, providing faster results. Mask R-CNN has been implemented in complement to image segmentation and object detection. A COCO pre-trained model was used to fine-tune our own model. 
Our key contribution is the first application of deep learning for classification of Monocytes and for Monocyte Index computation. The key significance of this work is that it may potentially save precious time in a life-saving procedure of blood transfusion. The broad significance is that similar methods of segmentation of multiple biological cells may be of great benefit to biological researchers that may want an automated method, as opposed to manual methods of classification and counting. With state-of-the-art deep learning models, we now have more accurate methods in this domain. 


In this project we will address semantic segmentation using MaskRCNN in images and video. The base repositories of this project are the following. The first one is the implementation for Tensorflow 1, and the second repository has the update for Tensorflow 2.

## Methods
We used Mask R-CNN techniques for instance segmentation which allows not only the detection and location of cells but also their classification and counting. 
## Datasets Acquisition and Processing
We use COCO pre-trained weights to train our own model.  Also, we use our own dataset collection image collected using ZEISS Axiocam 208 color/202 mono Microscope camera. A total of 500 image were divided in three folders (train, validation, and test).
## Proposed Solution
Mask-RCNN model has been modified to identify and count the monocytes present in the dataset. A minimum of 400 monocytes it must be identified in order to provide an acceptable Monocyte Index (MI). 
The formula to calculate the MI is as follows:
MI = ((Total # of Monocytes Ingested or Adhered RBC) / (Total Number of Monocytes Counted))𝑥 100%

The MI is the percentage of RBCs adhered, ingested or both (for the total) verses free monocytes. An MI of ‘Zero’ or 0 indicates there were no adhered or phagocytized red cells [1]. Experience with this procedure has been similar to others; in that MI values of ≤5% have indicated that incompatible blood can be given without the risk of an overt hemolytic transfusion reaction, but it does not guarantee normal long-term survival of those RBCs. MI values ranging from 5–20% have a reduced risk of clinical significance, but signs and symptoms of transfusion reaction may occur. Similarly, an MI of >20% indicates the antibody has clinical significance, which may range from abnormal RBC survival to clinically obvious adverse reactions.

## Results 
The neural network has two primary layers: coco transfer learning to speed up computation and enhance accuracy, and object identification and categorization, which supports the preceding layer. A Matterport Mask RCNN was used to identify objects. In the classification layer, a ResNet101 was used. Our training (fine-tuning) model took approximately 8 hours to complete the training with 20 epochs and 500 steps per epoch. Our GPU was a simple AMD Radeon R7 250. We had good quality (uncompressed) images. Our model was able to successfully detect, classify and count the two different classes assigned and the background. We used minimum confidence of 0.9, one GPU per image, and NMS threshold of 0.3, as our key parameters. The Confusion Matrix for the Test data is shown at Table 1.0.

![image](https://github.com/lapm93/MMA-Index-Calculation-Using-MaskRCNN/assets/100726201/77d214d4-304a-4ba1-808b-f718b76d72dc)

The time taken for detection, classification and printing the MI was about 5s. The false positives may be coming from the platelets, the noise in the image and the calibration of the microscope. A visual inspection was performed by a trained laboratory scientist with the results of our model. Compared to the trained laboratory scientist, the model can process with comparable accuracy. Human error can be as high as 30% at times of heavy workload and noisy images.

![image](https://github.com/lapm93/MMA-Index-Calculation-Using-MaskRCNN/assets/100726201/ce03f428-47bd-480c-8558-6580b1a3b191)
![image](https://github.com/lapm93/MMA-Index-Calculation-Using-MaskRCNN/assets/100726201/f0f7f00d-2955-4f04-9248-276e06edc47a)

## CONCLUSION AND RECOMMENDATIONS

Compared to a medical laboratory scientist, the model can process large amounts of data simultaneously, quickly, and efficiently, with approximately the same judgment accuracy as a human eye. The laboratory scientist can use the tool for the easy segmentations at high speed. For the uncertain segmentations, the manual methods may also be used for confirmation, which may significantly reduce the burden of the laboratory scientist and provide a useful reference for doctors to identify a potential blood candidate to be transfused. We hope that this method will increase the range of potential approaches to use a real time video analysis of Monocyte Index calculation.





Modifications were made to run detection using the webcam.

    https://github.com/matterport/Mask_RCNN
    https://github.com/akTwelve/Mask_RCNN

## Environment preparation

We will prepare an environment with python 3.7.7, Tensorflow 2.1.0 and keras 2.3.1

    $ conda create -n MaskRCNN anaconda python=3.7.7
    $ conda activate MaskRCNN
    $ conda install ipykernel
    $ python -m ipykernel install --user --name MaskRCNN --display-name "MaskRCNN"
    $ conda install tensorflow-gpu==2.1.0 cudatoolkit=10.1
    $ pip install tensorflow==2.1.0
    $ pip install jupyter
    $ pip install keras
    $ pip install numpy scipy Pillow cython matplotlib scikit-image opencv-python h5py imgaug IPython[all]
    
## Install MaskRCNN

    $ python setup.py install
    $ pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
    
## Try on Jupyter notebook

    $ cd samples
    $ jupyter notebook
    
## Test on console with images and video

### With Images

    $ cd samples
    $ python imagemask.py
    
### On video

    $ cd samples
    $ python videomask.py
    
# Training with custom-dataset
-   Label the data set with the tool [VIAv1.0](http://www.robots.ox.ac.uk/~vgg/software/via/via-1.0.0.html) (Do it with version 1.0.0)
-   Save validation and training data in folders named train and val
-   Save the annotations of the two data groups with the name: via_region_data.json
-   Run in google colab the file 

## Testing the trained model with custom-dataset

-   FOR SYSTEM TEST WITH IMAGES:
    
   Modify parameters
    
    -   model_filename = "mask_rcnn_0020.h5" # Here you must load the trained model with your dataset
    -   class_names = ['BG', 'FreeMonocytes', 'Ingested/AdheredMonocytes'] # The classes related to your BG model + custom classes
    -   min_confidence = 0.9 # Minimum level of confidence to accept a finding as positive
        
-   FOR VIDEO SYSTEM TEST:

    Modify parameters
    
    -  model_filename = "mask_rcnn_0020.h5" # Here you must load the trained model with your dataset
    -  class_names = ['BG', 'FreeMonocytes', 'Ingested/AdheredMonocytes'] # The classes related to your BG model + custom classes
    -   min_confidence = 0.9 # Minimum level of confidence to accept a finding as positive
    -   camera = cv2.VideoCapture(0) # If you want to run webcam
    -   camera = cv2.VideoCapture("video.mp4") # If you want to run a video uploading it from your PC
  
    


 
# Thanks

    Matterport, Inc
    https://github.com/matterport

    Adam Kelly
    https://github.com/akTwelve
    
    DavidReveloLuna
    https://github.com/DavidReveloLuna/MaskRCNN_Video

