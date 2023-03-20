docker run --gpus all -it --rm \
--mount type=bind,source=/home/linlongzhong/Projects/CenterPoint-origin/tensorrt/samples/centerpoint,target=/workspace/tensorrt/samples/centerpoint \
--mount type=bind,source=/home/linlongzhong/Projects/CenterPoint-origin/tensorrt/data/centerpoint,target=/workspace/tensorrt/data/centerpoint \
nvcr.io/nvidia/tensorrt:21.02-py3 /bin/bash