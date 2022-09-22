# Monocyte Monolayer Index calculation using MaskRCNN

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
    -   class_names = ['BG', 'helmet'] # The classes related to your BG model + custom classes
    -   min_confidence = 0.6 # Minimum level of confidence to accept a finding as positive
        
-   FOR VIDEO SYSTEM TEST:

    Modify parameters
    
    -  model_filename = "mask_rcnn_casco_0050.h5" # Here you must load the trained model with your dataset
    -  class_names = ['BG', 'helmet'] # The classes related to your BG model + custom classes
    -   min_confidence = 0.6 # Minimum level of confidence to accept a finding as positive
    -   camera = cv2.VideoCapture(0) # If you want to run webcam
    -   camera = cv2.VideoCapture("video.mp4") # If you want to run a video uploading it from your PC
  
    


 
# Thanks

    Matterport, Inc
    https://github.com/matterport

    Adam Kelly
    https://github.com/akTwelve


