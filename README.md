# SEVIO

**SEVIO** is a novel visual-inertial odemetry for a stereo event-based camera. Our visual module follow the work **[ESVO](https://sites.google.com/view/esvo-project-page/home)**, and fusion module based on ESKF.
#### video: https://youtu.be/IclaeypKIPc
#### paper: https://arxiv.org/abs/2303.05086

# 1. Installation

We have tested SEVIO on machines with the following configuration
* Ubuntu 18.04.5 LTS + ROS melodic + OpenCV 3.2

## 1.1 Driver Installation

To work with event cameras, especially for the Dynamic Vision Sensors (DVS/DAVIS), you need to install some drivers. Please follow the instructions (steps 1-9) at [rpg_dvs_ros](https://github.com/uzh-rpg/rpg_dvs_ros) before moving on to the next step. Note that you need to replace the name of the ROS distribution with the one installed on your computer.

## 1.2 Dependencies Installation

You should have created a catkin workspace in Section 1.1. If not, please go back and create one.

**Clone this repository** into the `src` folder of your catkin workspace.

	$ cd ~/catkin_ws/src 
	$ git clone https://github.com/WKunFeng/SEVIO.git

Dependencies are specified in the file [dependencies.yaml](dependencies.yaml). They can be installed with the following commands from the `src` folder of your catkin workspace:

	$ cd ~/catkin_ws/src
	$ sudo apt-get install python3-vcstool
	$ vcs-import < ESVO/dependencies.yaml

The previous command should clone the the repositories into folders called *catkin_simple*, *glog_catkin*, *gflags_catkin*, *minkindr*, etc. inside the `src` folder of your catking workspace, at the same level as this repository (SEVIO).

You may need `autoreconf` to compile glog_catkin. To install `autoreconf`, run
    
	$ sudo apt-get install autoconf


**yaml-cpp** is only used for loading calibration parameters from yaml files:

	$ cd ~/catkin_ws/src 
	$ git clone https://github.com/jbeder/yaml-cpp.git
	$ cd yaml-cpp
	$ mkdir build && cd build && cmake -DYAML_BUILD_SHARED_LIBS=ON ..
	$ make -j

Other ROS dependencies should have been installed in Section 1.1. 
If not by accident, install the missing ones accordingly.
Besides, you also need to have `OpenCV` (3.2 or later) and `Eigen 3` installed.

## 1.3 SEVIO Installation

After cloning this repository, as stated above (reminder)

	$ cd ~/catkin_ws/src 
	$ git clone https://github.com/WKunFeng/SEVIO.git
	
run

	$ catkin build sevio_time_surface sevio_core
	$ source ~/catkin_ws/devel/setup.bash


# 2. Usage

## 2.1 time_surface
This package implements a node that constantly updates the stereo time maps (i.e., time surfaces). To launch it independently, open a terminal and run the command:

    $ roslaunch sevio_time_surface stereo_time_surface.launch
    
To play a bag file, go to `sevio_time_surface/launch/rosbag_launcher` and modify the path in 
`[bag_name].launch` according to where your rosbag file is downloaded. Then execute

    $ roslaunch sevio_time_surface [bag_name].launch
    
## 2.2 full system

To launch the system, run

    $ roslaunch sevio_core system_xxx.launch

This will launch two *esvo_time_surface nodes* (for left and right event cameras, respectively). Then play the input bag file by running

    $ roslaunch sevio_time_surface [bag_name].launch


# 3. Notes



# 4. Datasets

The datasets we tesed can be downloaded from [sequences](https://drive.google.com/drive/folders/10HZ-sf0k96WDxMqyBHkDM14BgsFZoF45?usp=share_link).
