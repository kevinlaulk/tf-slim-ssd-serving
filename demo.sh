export PATH="/home/spaci/anaconda3/envs/lktf/bin:$PATH"
export PATH="/usr/local/cuda/bin":$PATH
export LD_LIBRARY_PATH="/usr/local/cuda/lib64":$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/spaci/anaconda3/envs/lktf/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=/home/spaci/anaconda3/envs/lktf/lib/libmkl_rt.so
export PATH="$PATH:$HOME/bin"
export LD_LIBRARY_PATH=/home/spaci/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/home/spaci/models/research:/home/spaci/models/research/slim

echo $PATH
echo $LD_LIBRARY_PATH
echo $LD_PRELOAD
echo $PYTHONPATH

python demo.py
