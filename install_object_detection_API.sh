pip3 uninstall tensorflow -y
pip3 uninstall keras -y 

pip3 install tensorflow==2.6.0
pip3 install keras==2.6.0

git clone https://github.com/tensorflow/models.git

cd models/research

# Compile protos.
sudo apt install protobuf-compiler -y
protoc object_detection/protos/*.proto --python_out=.

# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python3 -m pip install --use-feature=2020-resolver .


