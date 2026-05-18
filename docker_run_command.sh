docker run -itd \
    --name lcp \
    --gpus all \
    --entrypoint /bin/bash \
    -v /workingspace_aiclub/:/workingspace_aiclub/ \
    -v /AIClub_NAS/:/AIClub_NAS/ \
    -v /home/core_baotg/:/home/core_baotg/ \
    seemeai/llama-cpp-python:0.3.20 \
    -c "sleep infinity"
