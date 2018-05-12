if [ "${1}" != "" ]; then
    directory=`basename $0 .sh`
    prefix="nips/${directory}"

    mkdir -p $prefix

    python examples/mnist.py --prefix ${prefix}/mnist \
                             --model deepbn \
                             --l1_proj 50 \
                             --l1_train median \
                             --epochs 60 \
                             --starting_epsilon 0.01 \
                             --schedule_length 20 \
                             --batch_size 32 \
                             --verbose 400 \
                             --cuda_ids ${1}
else
    echo "Error: need to pass in GPU ids to run script on."
fi
