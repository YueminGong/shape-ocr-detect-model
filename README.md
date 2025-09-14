# shape-Text-Recognition-model-rpi5
Using a Raspberry Pi 5 with 8GB RAM and a CA1507 camera, equipped with the PaddleOCR algorithm (in a Paddle-Lite environment), enables the recognition of object dimensions and OCR code information. This solution can be applied as a universal model for recognizing and extracting text from all languages in a document.

based on the paddle-lite model
https://github.com/Qengineering/PaddleOCR-Lite-Document?tab=readme-ov-file

Follow the guild, the PaddleOCR model is verified by raspi-5 8GB

The critical code is located in Mydir/bin/release.



# dependencies
$ sudo apt-get update
$ sudo apt-get upgrade

$ sudo apt-get install -y libgtk2.0-dev libcanberra-gtk3-dev libgtk-3-dev libxvidcore-dev libx264-dev python3-dev python3-numpy python3-pip libtbbmalloc2 libtbb-dev libv4l-dev v4l-utils libopenblas-dev libatlas-base-dev libblas-dev liblapack-dev gfortran libhdf5-dev libprotobuf-dev libgoogle-glog-dev libgflags-dev protobuf-compiler

# opencv
only C++

$ sudo apt-get install libopencv-dev

Need Python also

$ sudo apt-get install python3-opencv

# model optimizer uses the pre-compiler model

opt_linux_aarch64



