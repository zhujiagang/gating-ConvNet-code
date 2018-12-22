
make clean
rm -rf build
mkdir build

#把之前编译的去掉

make all -j16
#make test -j16

#make runtest -j16
make pycaffe -j16
#make matcaffe -j16

#sh data/mnist/get_mnist.sh

#sh examples/mnist/create_mnist.sh
#sh examples/mnist/train_lenet.sh
