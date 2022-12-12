# Monocyte Monolayer Index calculation using MaskRCNN

Image segmentation has been used in the medical field for long time ago. Artificial intelligence algorithms have an incredible ability to learn and improve themselves. The Monocyte Monolayer Assay (MMA) is an in-vitro procedure that has been used for more than 40 years to predict in-vivo survivability of incompatible RBCs [1]. The MMA mimics extravascular hemolysis of red blood cells and may be used to guide transfusion recommendations. The Monocyte Index is a percentage of RBCs adhered, ingested or both by the monocytes. A minimum of 400 myocytes must be counted for enough data to determine the risk of transfusion of incompatible blood. In this work, Artificial Intelligence has been used to automate the Monocyte Index calculation, providing faster results. Mask R-CNN has been implemented in complement to image segmentation and object detection. A COCO pre-trained model was used to fine-tune our own model. 
Our key contribution is the first application of deep learning for classification of Monocytes and for Monocyte Index computation. The key significance of this work is that it may potentially save precious time in a life-saving procedure of blood transfusion. The broad significance is that similar methods of segmentation of multiple biological cells may be of great benefit to biological researchers that may want an automated method, as opposed to manual methods of classification and counting. With state-of-the-art deep learning models, we now have more accurate methods in this domain. 


In this project we will address semantic segmentation using MaskRCNN in images and video. The base repositories of this project are the following. The first one is the implementation for Tensorflow 1, and the second repository has the update for Tensorflow 2.
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
    
    -   model_filename = "mask_rcnn_casco_0050.h5" # Here you must load the trained model with your dataset
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

