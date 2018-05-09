if [ "${1}" != "" ]; then
    directory=`basename $0 .sh`
    prefix="nips/${directory}"

    mkdir -p $prefix

    python examples/cifar.py --prefix ${prefix}/cifar_exact ${args}

    for l1_proj in 10 50 100 150 200
    do
        python examples/cifar.py --prefix ${prefix}/cifar_${l1_proj} \
                                 --l1_proj ${l1_proj} \
                                 --l1_train median \
                                 --epochs 60 \
                                 --starting_epsilon 0.001 \
                                 --epsilon 0.031 \
                                 --schedule_length 20 \
                                 --verbose 400 \
                                 --cuda_ids ${1}
    done
else
    echo "Error: need to pass in GPU ids to run script on."
fi
