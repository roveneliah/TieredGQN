export JOB_NAME="job12"

gcloud ml-engine jobs submit training run46 \
  --job-dir gs://gqnmodel/  \
  --runtime-version 1.8 \
  --module-name training/cloud_train \
  --package-path training \
  --region us-central1


  # -- \
  # --train-files $TRAIN_DATA     \
  # --eval-files $EVAL_DATA

  # --train-steps 1000     \
  # --eval-steps 100     \
  # --verbosity DEBUG

  # --packages gqn.py,encoder.py,generator.py,data_reader.py \
