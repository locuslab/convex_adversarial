directory=`basename $0 .sh`
prefix="nips/${directory}"

mkdir -p $prefix

python examples/cifar.py --prefix ${prefix}/cifar \
                         --model vgg \
                         --l1_proj 50 \
                         --l1_train median \
                         --lr 0.001 \
                         --epsilon 0.031 \
                         --starting_epsilon 0.001 \
                         --cuda_ids 3 \
                         --verbose 200 \
                         --epochs 20 \
                         --batch_size 50 \
                         --schedule_length 20