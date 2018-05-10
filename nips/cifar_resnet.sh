if [ "${1}" != "" ] && [ "${2}" != "" ]; then
     directory=`basename $0 .sh`
     prefix="nips/${directory}_${2}"

     mkdir -p $prefix

     python examples/cifar.py --prefix ${prefix}/cifar \
                              --model resnet \
                              --l1_proj ${2} \
                              --l1_train median \
                              --l1_test median \
                              --lr 0.05 \
                              --epsilon 0.139 \
                              --starting_epsilon 0.001 \
                              --epochs 80 \
                              --batch_size 50 \
                              --schedule_length 20 \
                              --verbose 200 \
                              --cuda_ids ${1}
else
    echo "Error: need to pass in GPU ids and l1_proj to run script."
fi
