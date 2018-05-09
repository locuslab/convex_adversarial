if [ "${1}" != "" ]; then
    directory=`basename $0 .sh`
    prefix="nips/${directory}"
    args="--epochs 60 --starting_epsilon 0.01 --schedule_length 20 --verbose 200 --cuda_ids ${1}"

    mkdir -p $prefix

    python examples/mnist.py --prefix ${prefix}/mnist_exact ${args}

    for l1_proj in 10 50 100 150 200
    do
        python examples/mnist.py --prefix ${prefix}/mnist_${l1_proj} ${args} \
                                 --l1_proj ${l1_proj} \
                                 --l1_train median 
    done
else
    echo "Error: need to pass in GPU ids to run script on."
fi
