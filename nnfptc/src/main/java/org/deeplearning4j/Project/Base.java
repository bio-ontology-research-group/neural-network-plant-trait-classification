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

/**
 * Created by keiron on 02/08/15.
 */
public class Base {
    public static final Logger log = LoggerFactory.getLogger(Base.class);

    public static void main (String args []) throws Exception {
        final int numInputRows = 200;
        final int numInputCols = 200;
        final int numImages = 6001;
        final int numEpochs = 10;
        final int miniBatchSize = 50;

        log.info("Loading labels");
        String labeledPath = System.getProperty("user.home")+"/path/to/directory";
        List<String> labels = new ArrayList<>();
        for(File f : new File(labeledPath).listFiles()) {
            labels.add(f.getName());
        }

        log.info("Loading data");

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

        Evaluation evaluation = new Evaluation();
        DataSet testingData = splitTestAndTrain.getTest();
        INDArray predict = multiLayerNetwork.output(testingData.getFeatureMatrix());

        evaluation.eval(testingData.getLabels(), predict);



        log.info(evaluation.stats());
    }



}
