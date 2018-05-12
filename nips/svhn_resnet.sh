python examples/cifar.py --l1_proj 50 --l1_train median --l1_test median --starting_epsilon 0.001 --epochs 100 --schedule_length 40 --model resnet --cuda_ids 2,3 --prefix cifar10/resnet50_wide2 --verbose 400
if [ "${1}" != "" ] && [ "${2}" != "" ] && [ "${3}" != "" ] && [ "${4}" != "" ]; then
     directory=`basename $0 .sh`
     prefix="nips/${directory}"

     mkdir -p $prefix

     python examples/svhn.py --prefix ${prefix}/svhn${4}_group_${2}_wide_${3} \
                              --model resnet \
                              --resnet_N ${2} \
                              --resnet_factor ${3} \
                              --l1_proj ${4} \
                              --l1_train median \
                              --l1_test median \
                              --lr 0.05 \
                              --epsilon 0.001 \
                              --starting_epsilon 0.0001 \
                              --epochs 100 \
                              --batch_size 50 \
                              --schedule_length 40 \
                              --verbose 200 \
                              --cuda_ids ${1}
else
    echo "Error: need to pass in GPU ids, block size, width factor, and projection dimension to run script."
fi
