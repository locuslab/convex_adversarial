if [ "${1}" != "" ]; then
    directory=`basename $0 .sh`
    prefix="nips/${directory}"
    args="--epochs 60 --starting_epsilon 0.001 --schedule_length 20 --verbose 400 --cuda_ids ${1}"

    mkdir -p $prefix

    for factor in 1 2 4 8 16
    do
        python examples/svhn.py --prefix ${prefix}/svhn_${factor} ${args} \
                                 --model wide \
                                 --model_factor ${factor} \
                                 --l1_proj 50 \
                                 --l1_train median 
    done
else
    echo "Error: need to pass in GPU ids to run script on."
fi
