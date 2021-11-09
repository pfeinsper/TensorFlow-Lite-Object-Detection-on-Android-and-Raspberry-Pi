# get a fresh start (remember, the 64-bit OS is still under development)
sudo apt-get update
sudo apt-get upgrade
# install pip and pip3
sudo apt-get install python-pip python3-pip
# remove old versions, if not placed in a virtual environment (let pip search for them)
sudo pip uninstall tensorflow
sudo pip3 uninstall tensorflow
# utmost important: use only numpy version 1.19.5
# check the version first
pip3 list | grep numpy
# if not version 1.19.5, update!
sudo -H pip3 install numpy==1.19.5
# install the dependencies (if not already onboard)
sudo apt-get install gfortran
sudo apt-get install libhdf5-dev libc-ares-dev libeigen3-dev
sudo apt-get install libatlas-base-dev libopenblas-dev libblas-dev
sudo apt-get install liblapack-dev
# upgrade setuptools 40.8.1 -> 57.4.0
sudo -H pip3 install --upgrade setuptools
sudo -H pip3 install pybind11
sudo -H pip3 install Cython
# install h5py with Cython version 0.29.23 (± 15 min @1500 MHz)
sudo -H pip3 install h5py==3.1.0
# install gdown to download from Google drive
pip3 install gdown
# download the wheel
gdown https://drive.google.com/uc?id=1BLXP7RKEfTp9fxbmI8Qu2FdhU7NUxcwV
# install TensorFlow 2.6.0 (± 68 min @1500 MHz)
sudo -H pip3 install tensorflow-2.6.0-cp37-cp37m-linux_aarch64.whl
