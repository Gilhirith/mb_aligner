# mb_aligner
A Multi-beam affine stitching and 3d elastic registration library for running on Google Virtual Machines.

Uses python 3.4+.

# The Installation Note (Start from a newly created Google Virtual Machines):

sudo apt-get update

sudo apt-get install make build-essential git

sudo apt-get install python python-dev python3 python3-dev

sudo apt-get install htop

# Setup git

# If use virtualenv
wget https://bootstrap.pypa.io/get-pip.py

sudo python3 get-pip.py

sudo pip install --upgrade virtualenv

virtualenv --python python3 mbeam_aligner_venv

source ~/mbeam_aligner_venv/bin/activate

mkdir ~/Tools

cd ~/Tools

pip install intel-numpy

pip install intel-scipy

pip install intel-scikit-learn

pip install -U Cython

pip install --upgrade cython (update to neweat version)

# Install opencv3.1.0
sudo apt-get remove x264 libx264-dev

sudo apt-get install checkinstall cmake pkg-config yasm

sudo apt-get install libjpeg-dev libpng-dev

sudo apt-get install libtiff-dev

sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev

sudo apt-get install libxine2-dev libv4l-dev

sudo apt-get install qt5-default libgtk2.0-dev libtbb-dev

sudo apt-get install libatlas-base-dev

sudo apt-get install libvorbis-dev libxvidcore-dev

sudo apt-get install libopencore-amrnb-dev libopencore-amrwb-dev

sudo apt-get install x264 v4l-utils

sudo apt-get install libhdf5-dev

sudo apt-get install unzip

wget -O opencv-3.1.0.zip https://github.com/Itseez/opencv/archive/3.1.0.zip

unzip opencv-3.1.0.zip

wget -O opencv_contrib-3.1.0.zip https://github.com/Itseez/opencv_contrib/archive/3.1.0.zip

unzip opencv_contrib-3.1.0.zip

cd opencv-3.1.0

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B831c8acWYXhVlFDdDRva0FzX19aa3dlUnlKeXZMeUJwX3Vn' -O ./modules/python/common.cmake

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B831c8acWYXhU0xxemNwdXhCYXh0VG9Fdjg1NEo5OGVjbkNN' -O ../opencv_contrib-3.1.0/modules/tracking/include/opencv2/tracking/onlineMIL.hpp

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B831c8acWYXhMFBMbzRqZjNRNW55YWVEeDZOVGttMVhwdVlF' -O ../opencv_contrib-3.1.0/modules/tracking/src/onlineMIL.cpp

mkdir build

cd build

mkdir ~/Tools/opencv_install-3.1.0

cmake -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.1.0/modules -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=../../opencv_install-3.1.0 -D PYTHON3_EXECUTABLE=~/mbeam_aligner_venv/bin/python -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D ENABLE_PRECOMPILED_HEADERS=OFF -D WITH_TBB=ON -D WITH_QT=ON -D WITH_OPENGL=ON  ..

make -j4

make install

ln -s ~/Tools/opencv_install-3.1.0/lib/python3.5/site-packages/cv2.cpython-35m-x86_64-linux-gnu.so ~/mbeam_aligner_venv/lib/python3.5/site-packages/cv2.so

*add the following 3 sentences into environment.sh

export PKG_CONFIG_PATH=~/Tools/opencv_install-3.1.0/lib/pkgconfig/:$PKG_CONFIG_PATH

export LD_LIBRARY_PATH=~/Tools/opencv-3.1.0/3rdparty/ippicv/unpack/ippicv_lnx/lib/intel64/:~/Tools/opencv_install-3.1.0/lib/:$LD_LIBRARY_PATH

export LIBRARY_PATH=~/Tools/opencv-3.1.0/3rdparty/ippicv/unpack/ippicv_lnx/lib/intel64/:~/Tools/opencv_install-3.1.0/lib/:$LIBRARY_PATH

source environment.sh

# Set up new disk
sudo lsblk

sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb

sudo mkdir -p /mnt/disks/data_disk

sudo mount -o discard,defaults /dev/sdb /mnt/disks/data_disk

*Add permission

sudo chmod a+rwx /mnt/disks/data_disk/

# Set up automatic mounting
sudo cp /etc/fstab /etc/fstab.backup

sudo blkid /dev/sdb

echo UUID=`sudo blkid -s UUID -o value /dev/sdb` /mnt/disks/data_disk ext4 discard,defaults,nofail 0 2 | sudo tee -a /etc/fstab

# Install mb_aligner and rh_renderer (you need to change sources for those on code.harvard.edu)
git clone https://github.com/adisuissa/rh_img_access_layer.git

git clone https://github.com/Gilhirith/mb_aligner_SH.git

git clone https://github.com/adisuissa/gcsfs.git

git clone https://github.com/Rhoana/rh_config.git

git clone https://github.com/Rhoana/rh_logger.git

git clone https://github.com/Rhoana/tinyr.git

git clone -b google_cloud https://github.com/Rhoana/rh_renderer.git


cd rh_img_access_layer

pip install -e .

cd gcsfs

pip install -e .

cd tinyr

pip install -r requirements.txt

pip install -e .

cd rh_config

pip install -e .

cd rh_logger

pip install -e .

cd rh_renderer

pip install -e .

cd mb_aligner_SH

pip install -e .




# Command to run stitching
#!/bin/bash

python -u ./mb_aligner/scripts/2d_stitch_driver.py --ts_dir /mnt/disks/data_disk/Primate/tiles -c conf_Susan_C1_test4.yaml -o /mnt/disks/data_disk/Primate/2d_output_dir_sift


# Command to run rendering
#!/bin/bash

python -u ./mb_aligner/scripts/3d_render_driver.py /mnt/disks/data_disk/Primate/2d_output_dir_sift/Sec009 -o /mnt/disks/data_disk/Primate/2d_render_full_res_4k_tiles --scale 1 --tile_size 4096 --invert_image -p 33
