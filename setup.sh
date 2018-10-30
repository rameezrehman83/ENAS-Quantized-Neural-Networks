
# Run the below commands to install pip and virtualenv 

# sudo apt-get install python3-pip
# sudo pip3 install virtualenv 

virtualenv -p python3 enas_env
source enas_env/bin/activate 
pip install tensorflow-gpu
pip install matplotlib
pip install keras 
# Replace the base_layer.py file
mv keras/base_layer.py enas_env/lib/python3.5/site-packages/keras/engine/base_layer.py
