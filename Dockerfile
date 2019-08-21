FROM ubuntu:18.04

# `docker build -t mkldnn .`  (add `--no-cache` for a full rebuild)

RUN apt-get update -y
RUN apt-get install -y git curl cmake python3-dev python3-pip
RUN git clone --recursive https://github.com/pytorch/pytorch.git
WORKDIR pytorch/
RUN git remote add gujinghui https://github.com/gujinghui/pytorch.git && git fetch gujinghui && git checkout gujinghui/mkldnn_v0.20.1 && git submodule update --init --recursive
WORKDIR ..
RUN python3 -m pip install --verbose --user virtualenv
RUN python3 -m pip --version
RUN python3 -m virtualenv -p python3 pytorch_virtualenv
RUN . ./pytorch_virtualenv/bin/activate
RUN python3 -m pip install --verbose --force-reinstall future leveldb numpy protobuf pydot python-gflags pyyaml scikit-image setuptools six hypothesis typing tqdm
# Upgrade pip to latest before building pytorch
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --verbose ./pytorch/.
RUN python3 -m pip install --verbose --force-reinstall mkl
#RUN python3 -c "import pydoc; pydoc.locate('torch.nn.modules.conv.Conv1d')"
#RUN python3 -c "import numpy; print(numpy.random.__file__)"
RUN ls -l /usr/local/lib/python3.6/dist-packages/torch/lib
RUN ldd /usr/local/lib/python3.6/dist-packages/torch/_C.cpython-36m-x86_64-linux-gnu.so
RUN readelf -d /usr/local/lib/python3.6/dist-packages/torch/_C.cpython-36m-x86_64-linux-gnu.so
RUN ldd /usr/local/lib/python3.6/dist-packages/torch/lib/libtorch.so
RUN readelf -d /usr/local/lib/python3.6/dist-packages/torch/lib/libtorch.so
#RUN ldd /usr/local/lib/python3.6/dist-packages/torch/lib/libiomp5.so
#RUN readelf -d /usr/local/lib/python3.6/dist-packages/torch/lib/libiomp5.so
#RUN ldd /usr/local/lib/python3.6/dist-packages/torch/lib/libmklml_intel.so
#RUN readelf -d /usr/local/lib/python3.6/dist-packages/torch/lib/libmklml_intel.so
#RUN file /usr/local/lib/python3.6/dist-packages/torch/lib/libmklml_intel.so
#RUN file /usr/local/lib/python3.6/dist-packages/torch/lib/libiomp5.so

# Did we get a new version?
RUN python3 -m pip --version
# To get a good overview of the library search path
#RUN LD_DEBUG=libs ldd /usr/local/lib/python3.6/dist-packages/torch/_C.cpython-36m-x86_64-linux-gnu.so
#RUN LD_DEBUG=libs ldd /usr/local/lib/python3.6/dist-packages/torch/lib/libmklml_intel.so

RUN python3 -c 'from torch._C import *'

RUN python3 -c 'import torch; print(torch.__config__.show())'

RUN python3 ./pytorch/test/test_nn.py

# The `ldd .../libmklml_intel.so` shows it doesn't find the `libiomp5.so`
# that's right next to it.
# The `readelf -d` shows that it has no rpath set.  It needs `$ORIGIN`.

# Note: split Dockerfile in two, to only avoid cache for pytorch build and next
# steps?
