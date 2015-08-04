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



