	# neural-network-plant-trait-classification

## Installation
Getting this project set up is a fairly simple task that will only take a few minutes of your time.

1. [Maven](https://maven.apache.org/) is a "software project management and comprehension tool". Most Java developers already have this installed on their system, so to check try the ```mvn --version``` command. If it's not installed then, install it using the following command. ```sudo apt-get install maven```.
2. As well as being one of Miguel's most hated software packages, [git](https://git-scm.com/) is a distributed revision control system that is required to set up most of the packages required to get started. Installation is fairly simple, just run ```sudo apt-get install git``` - and you should be good to go!
3. [jblas](http://jblas.org/) is a "fast linear algebra library" for Java. As it's being used for CPUs, the installation of native bindings are required. This can be done in Ubuntu with the following command. ```git clone https://github.com/mikiobraun/jblas.git ; cd jblas ; mvn install```.
4. This project uses the [Deeplearning4j](https://github.com/deeplearning4j/deeplearning4j) deep learning framework comprehensively. A handy script can be found in ```scripts/setup.sh``` to automate the installation process of the framework.

After that, load the ```nnfptc``` project into IntelliJ, or whatever IDE you use and everything should be set up. If not, feel free to [email me](mailto:keo7@aber.ac.uk) or create a [new issue](https://github.com/bio-ontology-research-group/neural-network-plant-trait-classification/issues).

## Traits

1. Life Form
2. Growth Form
3. Leaf - Structure
4. Leaf - Form
5. Leaf Venation
6. Flower - Inflorescence
7. Flower - Symmetry
8. Flower - Colour
9. Flower - Petal (Tepal) number
10. Flower - Stamen number
11. Fruit - Type
12. Fruit - Pericarp
13. Important diagnostics
14. Fruit - Dehiscence
15. Fruit - Surface
16. Leaf - Margin


## Fanf
'''
Using gpu device 0: Quadro K2200
X_train shape: (14850, 3, 64, 64)
Y_train shape: (14850, 2)
train samples: 14850 validation samples: 1648
Epoch  0
Training
/usr/local/lib/python2.7/dist-packages/keras/models.py:339: UserWarning: The "train" method is deprecated, use "train_on_batch" instead.
  warnings.warn('The "train" method is deprecated, use "train_on_batch" instead.')
14800/14850 [============================>.] - ETA: 0s - train loss: 0.6655 - train accuracy:: 0.6597Valid
1648/1648 [==============================] - 6s     
Epoch  1
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.5076 - train accuracy:: 0.7493Valid
1648/1648 [==============================] - 6s     
Epoch  2
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.4636 - train accuracy:: 0.7788Valid
1648/1648 [==============================] - 6s     
Epoch  3
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.4314 - train accuracy:: 0.7961Valid
1648/1648 [==============================] - 6s     
Epoch  4
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.4052 - train accuracy:: 0.8139Valid
1648/1648 [==============================] - 6s     
Epoch  5
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.3764 - train accuracy:: 0.8318Valid
1648/1648 [==============================] - 6s     
Epoch  6
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.3362 - train accuracy:: 0.8502Valid
1648/1648 [==============================] - 5s     
Epoch  7
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.3024 - train accuracy:: 0.8694Valid
1648/1648 [==============================] - 5s      
Epoch  8
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.2528 - train accuracy:: 0.8923Valid
1648/1648 [==============================] - 5s     
Epoch  9
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.2214 - train accuracy:: 0.9103Valid
1648/1648 [==============================] - 5s     
Epoch  10
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.1786 - train accuracy:: 0.9331Valid
1648/1648 [==============================] - 6s     
Epoch  11
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.1419 - train accuracy:: 0.9447Valid
1648/1648 [==============================] - 5s     
Epoch  12
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.1160 - train accuracy:: 0.9572Valid
1648/1648 [==============================] - 6s     
Epoch  13
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.1191 - train accuracy:: 0.9556Valid
1648/1648 [==============================] - 5s     
Epoch  14
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0880 - train accuracy:: 0.9678Valid
1648/1648 [==============================] - 5s      
Epoch  15
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0762 - train accuracy:: 0.9725Valid
1648/1648 [==============================] - 5s     
Epoch  16
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0631 - train accuracy:: 0.9778Valid
1648/1648 [==============================] - 5s     
Epoch  17
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0602 - train accuracy:: 0.9776Valid
1648/1648 [==============================] - 5s     
Epoch  18
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0522 - train accuracy:: 0.9823Valid
1648/1648 [==============================] - 5s      
Epoch  19
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0445 - train accuracy:: 0.9844Valid
1648/1648 [==============================] - 5s     
Epoch  20
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0450 - train accuracy:: 0.9852Valid
1648/1648 [==============================] - 5s     
Epoch  21
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0380 - train accuracy:: 0.9864Valid
1648/1648 [==============================] - 5s     
Epoch  22
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0339 - train accuracy:: 0.9877Valid
1648/1648 [==============================] - 5s     
Epoch  23
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0346 - train accuracy:: 0.9878Valid
1648/1648 [==============================] - 6s     
Epoch  24
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0364 - train accuracy:: 0.9866Valid
1648/1648 [==============================] - 6s     
Epoch  25
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0321 - train accuracy:: 0.9881Valid
1648/1648 [==============================] - 6s     
Epoch  26
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0301 - train accuracy:: 0.9901Valid
1648/1648 [==============================] - 5s     
Epoch  27
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0253 - train accuracy:: 0.9907Valid
1648/1648 [==============================] - 5s     
Epoch  28
Training
14800/14850 [============================>.] - ETA: 0s - train loss: 0.0293 - train accuracy:: 0.9907Valid
1648/1648 [==============================] - 5s     
'''
