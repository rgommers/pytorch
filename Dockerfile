FROM ubuntu:18.04

# `docker build -t mkldnn .`  (add `--no-cache` for a full rebuild)

RUN apt-get update -y
RUN apt-get install -y git curl cmake python3-dev python3-pip
RUN git clone --recursive https://github.com/pytorch/pytorch.git
RUN python3 -m pip install --verbose --user virtualenv
RUN python3 -m virtualenv -p python3 pytorch_virtualenv
RUN . ./pytorch_virtualenv/bin/activate
RUN python3 -m pip install --verbose --force-reinstall future leveldb numpy protobuf pydot python-gflags pyyaml scikit-image setuptools six hypothesis typing tqdm
RUN python3 -m pip install --verbose --force-reinstall mkl
RUN python3 -m pip install --verbose ./pytorch/.
RUN python3 -c "import pydoc; pydoc.locate('torch.nn.modules.conv.Conv1d')"
RUN python3 -c "import numpy; print(numpy.random.__file__)"
RUN ls /usr/local/lib/python3.6/dist-packages/torch/lib
RUN ldd /usr/local/lib/python3.6/dist-packages/torch/_C.cpython-36m-x86_64-linux-gnu.so
RUN ldd /usr/local/lib/python3.6/dist-packages/torch/_C.cpython-36m-x86_64-linux-gnu.so
RUN readelf -d /usr/local/lib/python3.6/dist-packages/torch/_C.cpython-36m-x86_64-linux-gnu.so
RUN readelf -d /usr/local/lib/python3.6/dist-packages/torch/lib/libtorch.so
RUN readelf -d /usr/local/lib/python3.6/dist-packages/torch/lib/libiomp5.so
RUN readelf -d /usr/local/lib/python3.6/dist-packages/torch/lib/libmklml_intel.so
RUN ldd /usr/local/lib/python3.6/dist-packages/torch/lib/libmklml_intel.so
#RUN python3 ./pytorch/test/test_nn.py
RUN deactivate
