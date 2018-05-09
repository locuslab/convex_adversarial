if [ "${1}" != "" ]; then
    directory=`basename $0 .sh`
    prefix="nips/${directory}"

    mkdir -p $prefix

    for factor in 1 2 3 4 5
    do
        python examples/cifar.py --prefix ${prefix}/cifar_${factor} \
                                 --model deep \
                                 --model_factor ${factor} \
                                 --l1_proj 50 \
                                 --l1_train median \
                                 --epochs 60 \
                                 --starting_epsilon 0.001 \
                                 --epsilon 0.031 \
                                 --schedule_length 20 \
                                 --batch_size 32 \
                                 --verbose 400 \
                                 --cuda_ids ${1}
    done
else
    echo "Error: need to pass in GPU ids to run script on."
fi
