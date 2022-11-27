sudo apt-get update
sudo apt-get install build-essential
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
wget https://www.python.org/ftp/python/3.7.13/Python-3.7.13.tgz
tar -xf Python-3.7.*.tgz
cd Python-3.7.*/
./configure --enable-optimizations
make -j $(nproc)
sudo make altinstall
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
/usr/local/bin/python3.7 get-pip.py
/usr/local/bin/python3.7 -m pip install --upgrade pip
/usr/local/bin/python3.7 -m pip install virtualenv
cd ..
/usr/local/bin/python3.7 -m virtualenv -p /usr/local/bin/python3.7 venv
git clone https://github.com/AndrewJudson/mesh-transformer-jax.git
source venv/bin/activate
cd mesh-transformer-jax
pip install -r requirements.txt
pip uninstall jax jaxlib
#pip install "jax[tpu]==0.2.12" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html # broken on TPU right now. check jax.device_count
pip install "jax[tpu]==0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install jaxlib==0.1.68
# Need to install TPU specific version of jax==0.2.12?
TPU_LIBRARY_PATH=/home/andrew/venv/libtpu.so/libtpu.so python device_train.py --config=elon.json --tune-model-path=gs://cama-finetuning/step_383500/

#tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such fileor directory
#2022-11-23 19:41:55.587534: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: :/usr/local/lib
#2022-11-23 19:41:55.587586: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)

python to_hf_weights.py --input-ckpt gs://cama-finetuning/mesh_jax_pile_6B_rotary/step_1 --config ./configs/6B_roto_256.json --output-path gs://cama-finetuning/gpt-j-6B --cpu --dtype fp32