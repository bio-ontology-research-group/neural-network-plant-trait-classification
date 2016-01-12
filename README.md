# Semantic Fine Grained Plant Trait Classification

*Update*: Everything before was awful and terrible, so I needed to fix it.

## Installation
Getting this project set up is a fairly simple task that will only take a few minutes of your time.

The project is now built on the [keras](https://github.com/fchollet/keras) theano-based deep learning library. It's built entirely on simplicity and ease of implementation.

I found the easiest way to set this up was via pip. pip is a package management system for the python programming language.

To install pip on a Ubuntu system, simply run the following command in the command line.

```bash
sudo apt-get install python-pip python-dev build-essential
```

**Note:** This project was made using Python **2.7.6**.

Now that this is set up, you're going to need to install a few dependencies first.

1. **numpy and scipy:** ```sudo pip install scipy```
2. **PIL:** ```sudo pip install pillow```
3. **pyaml:** ```sudo pip install pyyaml```
4. **theano:** ```sudo pip install theano```

And finally, you can install keras using the following command.

```
sudo pip install keras
```

After that, load the ```software/neuralNetworkSoftware``` project into Pycharm, or whatever IDE/plain text editor you use and everything should be set up. If not, feel free to [email me](mailto:keo7@aber.ac.uk) or create a [new issue](https://github.com/bio-ontology-research-group/neural-network-plant-trait-classification/issues).

## Running on a GPU

Provided that you have CUDA set up correctly, running tests on a GPU is as easy as running a line in terminal.

```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 filename.py
```