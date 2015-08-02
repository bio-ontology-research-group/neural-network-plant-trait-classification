package org.deeplearning4j.Project;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.slf4j.Logger;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


/*
 * Base.java
 *
 * Base java file for work. Use this as a template for experimentation.
 */

public class Base {
    public static final Logger log = LoggerFactory.getLogger(Base.class);

    public static void main (String args []) throws Exception {

        // Parameters for loading data.
        String labeledPath = System.getProperty("user.home")+"/path/to/directory";
        int numInputRows = 200;
        int numInputCols = 200;
        int numImages = 6001;
        List<String> labels = new ArrayList<>();

        // Parameters for training/evaluation
        final int numEpochs = 10;
        final int miniBatchSize = 50;
        Evaluation evaluation = new Evaluation();

        log.info("Loading labels");
        for(File f : new File(labeledPath).listFiles()) {
            labels.add(f.getName());
        }

        log.info("Loading data");

        /*
         * RecordReader: class from Canova (https://github.com/jpatanooga/Canova)
         * that converts byte input that's orientated to a collection of elements that
         * are a FIXED NUMBER and indexed with a UID. This process is requried to
         * vectorise the data, of which every element is a feature.
         *
         * ImageRecordReader: a subclass of RecordREader and automatically takes
         * 28x28 dimension images. We need to change the dimensions of this to match
         * our images, of which we do in the 3rd parameter. We also feed in the labels
         * to validate the neural net model results (the last parameter identifies that).
         *
         * DataSetIterator: a class that traverses through the RecordReader, and helps
         * keep track of how many and what images have already been fed through the
         * model. In this line, we are also converting the images into vectors of elements,
         * opposed to a 28x28 matrix, and identify the number of labels.
         */

        RecordReader recordReader = new ImageRecordReader(numInputRows, numInputCols, true);
        recordReader.initialize(new FileSplit(new File(labeledPath)));
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, numImages, numInputCols * numInputCols, labels.size());
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        log.info("Building model");

        MultiLayerConfiguration multiLayerConfiguration = new NeuralNetConfiguration.Builder().build();

        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(multiLayerConfiguration);
        multiLayerNetwork.init();

        multiLayerNetwork.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(1)));

        log.info("Splitting dataset");
        DataSet allData =  dataSetIterator.next();

        SplitTestAndTrain splitTestAndTrain = allData.splitTestAndTrain(0.9);
        DataSet trainingData = splitTestAndTrain.getTrain();

        log.info("Training on:");

        List<DataSet> trainingList = trainingData.asList();
        for(int i = 0 ; i < numEpochs ; i++) {
            Collections.shuffle(trainingList);
            DataSetIterator trainingListIterator = new ListDataSetIterator(trainingList, miniBatchSize);
            multiLayerNetwork.fit(trainingListIterator);
        }

        log.info("Evaluting model");

        DataSet testingData = splitTestAndTrain.getTest();
        INDArray predict = multiLayerNetwork.output(testingData.getFeatureMatrix());

        evaluation.eval(testingData.getLabels(), predict);

        log.info(evaluation.stats());
    }



}
