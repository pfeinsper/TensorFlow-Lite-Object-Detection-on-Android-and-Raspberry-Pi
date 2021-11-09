pip3 uninstall tensorflow -y
pip3 install tensorflow==2.6.0

pip3 uninstall keras -y 
pip3 install keras==2.6.0


git clone https://github.com/tensorflow/models.git

cd models/research

git clone https://github.com/cocodataset/cocoapi.git

cd cocoapi/PythonAPI
python setup.py build_ext --inplace
python setup.py build_ext install
cd ../..

cp -r pycocotools .

# Compile protos.
sudo apt install protobuf-compiler -y
protoc object_detection/protos/*.proto --python_out=.

# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python3 -m pip install --use-feature=2020-resolver .

python3 object_detection/builders/model_builder_tf2_test.py
