# Multi-Purpose Depth Pipeline
This repo contains the final project for ECGR 4090: Real-Time Machine Learning for Nathan Hewitt, Keith Chang, and Bradley Matheson. This represents a custom pipeline that has interchangeable components for multiple use cases. Depth estimation is performed on an image, while the image is segmented to identify either a round target or people. These results are then used to either aim a turret at a target, or calculate distances between people. 

## Required Packages
This repo runs on Python 3.6. The required packages are found in requirements.txt

## Setup

### Environment Prep
If using Anaconda, first create a new environment, activate it, and add Anaconda packages. 
**Anaconda is highly recommended** so that packages installed for other projects do not interfere with this installation, or vice versa.
If not using Anaconda, just skip these three lines.<br>
`conda create -n depth python=3.6`<br>
`conda activate depth`<br>
`conda install anaconda`

Install required packages with pip<br>
`pip install -r requirements.txt`<br>
(This may have missed a few, since I generated it automatically. If you come across missing ones, please add them to requirements.txt)

### EfficientHRNet Configuration
This is optional: without EHRNet, you can use the torchvision ResNet segmentor instead whether or not EHRNet is installed. You just will need to butcher all the parts of the code that import EHRNet if you choose not to install it.

I don't know if EfficientHRNet's segmentation implementation has been open-sourced, so it's not included in this repo. There is a placeholder for it at `src/human_seg/EfficientSegNet`. Simply copy in the EfficientSegNet code there, **and copy the seg_config.yaml to that folder**.  Modify the config as needed. I feel like there's another line somewhere that needs to be changed, but can't find it now-- please update this README if you run into an issue and solve it.

### Calibration
Make a folder somewhere holding only your 9x6 OpenCV checkerboard calibration images. Then, run `python src/calibrate.py /the/path/to/your/checkerboard/imgs/*`. This should create a pickle file encoding your camera's calibration. Make sure it is saved in the `src` directory.

### Run Test
To run this on one image, modify the first line of `main` in `run_pipeline.py` (I've been too busy to parametrize this yet). By default, it should segment for people. A visualization will appear with each person's pixels placed in 3D space, a coordinate frame at the camera's origin, and a colored sphere marking the center point of each person. You can use the mouse to pan and inspect it. Press q or the close button once done. 

The console should output an n*n matrix, where n is the count of people in the image. Spaces [i,i] represent the ith person's distance from the camera, while spaces [i,j] (for i != j) represent the distance between person i and person j.

It's usually clear which person has which index based on their distance from the camera, but if you're unsure, the visualization can help. The order of circles is always red, green, blue, etc (see `src/Projector.py` for colors of more than 3 people), so the person with a red sphere is person 0, with a green sphere is person 1, etc.
