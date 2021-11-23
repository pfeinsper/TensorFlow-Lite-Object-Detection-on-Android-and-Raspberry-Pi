sudo apt update
sudo apt upgrade

pip3 uninstall tensorflow -y 
pip3 install tensorflow==2.6.0


git clone https://github.com/tensorflow/models.git

cd models/research

pip3 install cython
pip3 install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

# Compile protos.
sudo apt install protobuf-compiler -y
protoc object_detection/protos/*.proto --python_out=.

# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
pip3 install .

python3 object_detection/builders/model_builder_tf2_test.py
