gcloud compute tpus tpu-vm create tpu-name \
--zone us-central1-a \
--accelerator-type v3-8 \
--version tpu-vm-base

gcloud compute tpus tpu-vm ssh tpu-name --zone us-central1-a