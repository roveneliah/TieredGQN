# export CLOUD_CONFIG = training/cloudml-gpu.yaml

gcloud ml-engine jobs submit training run61 \
  --job-dir gs://gqnmodel/  \
  --runtime-version 1.8 \
  --module-name cloud_train \
  --package-path model \
  --region us-central1 \
  --config=cloudml-gpu.yaml


  # -- \
  # --train-files $TRAIN_DATA     \
  # --eval-files $EVAL_DATA

  # --train-steps 1000     \
  # --eval-steps 100     \
  # --verbosity DEBUG

  # --packages gqn.py,encoder.py,generator.py,data_reader.py \
