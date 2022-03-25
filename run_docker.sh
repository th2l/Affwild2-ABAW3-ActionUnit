docker run --gpus all --ipc=host -it --rm \
        --user $UID:$GID \
        --volume="/etc/group:/etc/group:ro" \
        --volume="/etc/passwd:/etc/passwd:ro" \
        --volume="/etc/shadow:/etc/shadow:ro" \
        -v /home/hvthong/sXProject/Affwild2_ABAW3:/home/hvthong/sXProject/Affwild2_ABAW3 \
        -v /mnt/Work/Dataset/Affwild2_ABAW3:/mnt/Work/Dataset/Affwild2_ABAW3 \
        -w /home/hvthong/sXProject/Affwild2_ABAW3 \
        affwild2/pytorch:1.11-cuda11.3 bash scripts/train_AU.sh