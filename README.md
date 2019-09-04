# Class Activation and Attention Models for Face Classification

## Description of contents

`data_preprocessing/`: contains the Python scripts needed to tag the images by class (smiling/not smiling, male/female) and upload both the images and the metadata to the AWS S3 bucket.

`CNN Research Notebook.ipynb`: Jupyter Notebook that reflects setting up basic vanilla CNN image classification architectures.

`Image Tagging and Processing.ipynb`: Jupyter Notebook that was used to develop the scripts for tagging each image in the FEI dataset by their gender.

`MNIST Class Activation Heatmap Example.ipynb`: contains the implementation for Class Activation Heatmaps on the MNIST dataset.

`vanilla_cnn_faces.py`: contains an implementation of VGG16 CNN architecture that achieves strong performance on gender classification of FEI dataset. This was run on a remote GPU-optimized EC2 instance.

`Class Activation Map Faces.ipynb`: contains VGG16 CAM implementation on the FEI dataset, along with some results and heatmap examples.

`Vanilla Self Attention.ipynb`: contains VGG16 Multi-Head Augmented Attention model implementation on the FEI dataset. Note that I have annotated which portions of the code are borrowed from another Github repository.
