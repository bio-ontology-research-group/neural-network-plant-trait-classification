package org.deeplearning4j.Project.FloweringAndNotFlowering;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by osheak on 03/08/15.
 */
public class FloweringAndNotFloweringSmallScale {
    public static final Logger log = LoggerFactory.getLogger(FloweringAndNotFloweringSmallScale.class);

    public static void main (String args[]) throws Exception {
        String labeledPath = System.getProperty("user.home")+"/images/fanfs/";
        int numInputRows = 350;
        int numInputCols = 350;
        int numImages = 2000;
        List<String> labels = new ArrayList<>();

        int numInput = numInputRows * numInputCols;
        int numOut = labels.size();

        final int numEpochs = 10;
        final int miniBatchSize = 50;
        Evaluation evaluation = new Evaluation();

        log.info("Loading labels");
        for(File f : new File(labeledPath).listFiles()) {
            labels.add(f.getName());
        }

        log.info("Loading data");

        RecordReader recordReader = new ImageRecordReader(numInputRows, numInputCols, true);
        recordReader.initialize(new FileSplit(new File(labeledPath)));
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, numImages, numInput, numOut);
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        log.info("Building model");

        // A Bernoulli RBM
        MultiLayerConfiguration multiLayerConfiguration = new NeuralNetConfiguration.Builder()
                .layer(new RBM())
                .nIn(numInput).nOut(numOut)
                .learningRate(0.06)
                .iterations(20)
                .visibleUnit(RBM.VisibleUnit.BINARY).hiddenUnit(RBM.HiddenUnit.BINARY)
                .list(8).hiddenLayerSizes(1500,1250,1000,850,600,450,200,numOut).build();

        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(multiLayerConfiguration);
        multiLayerNetwork.init();

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
