# export CLOUD_CONFIG = training/cloudml-gpu.yaml

gcloud ml-engine jobs submit training run60 \
  --job-dir gs://gqnmodel/  \
  --runtime-version 1.8 \
  --module-name training/cloud_train \
  --package-path training \
  --region us-central1 \
  --config=training/cloudml-gpu.yaml


  # -- \
  # --train-files $TRAIN_DATA     \
  # --eval-files $EVAL_DATA

  # --train-steps 1000     \
  # --eval-steps 100     \
  # --verbosity DEBUG

  # --packages gqn.py,encoder.py,generator.py,data_reader.py \
