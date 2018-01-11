#block(name=[net], threads=3, memory=10000, subtasks=1, gpus=1, hours=10000)
  echo "Using python 3, so we need to export where the custom opencv3 install is:"
  export PYTHONPATH=/mylibs/lib/python3.5/dist-packages/
  echo $PYTHONPATH
  echo "Starting net training job"
  cd /cache/andres/bonnet/train_py
  echo ""
  echo "--------------------------------------------------------------------"
  echo "cfg/net.yaml"
  cat cfg/net.yaml
  echo ""
  echo "--------------------------------------------------------------------"
  echo "cfg/data.yaml"
  less cfg/data.yaml
  echo ""
  echo "--------------------------------------------------------------------"
  echo "cfg/train.yaml"
  less cfg/train.yaml
  echo ""
  echo "--------------------------------------------------------------------"
  echo "running program as ./cnn_train.py -d cfg/data.yaml -n cfg/net.yaml -t cfg/train.yaml -l ~/logs/temp/"
  ./cnn_train.py -d cfg/data.yaml -n cfg/net.yaml -t cfg/train.yaml -l ~/logs/temp/