package org.deeplearning4j.Project.Fanf;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.weights.WeightInit;
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
        String labeledPath = System.getProperty("user.home")+"/datasets/fanfSmall/";
        int numInputRows = 28;
        int numInputCols = 28;
        int numImages = 200;
        List<String> labels = new ArrayList<>();

        int numInput = numInputRows * numInputCols;
        int iterations = 5;
        int seed = 123;

        final int numEpochs = 20;
        final int miniBatchSize = 10;
        int numOut = labels.size();

        Evaluation evaluation = new Evaluation(numOut);

        log.info("Loading labels");
        for(File f : new File(labeledPath).listFiles()) {
            labels.add(f.getName());
        }



        log.info("Loading data");

        RecordReader recordReader = new ImageRecordReader(numInputRows, numInputCols, true);
        recordReader.initialize(new FileSplit(new File(labeledPath)));
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, numImages, numInput, labels.size());
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        log.info("Building model");


        log.info("Build model....");
        MultiLayerConfiguration multiLayerConfiguration = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.NORMALIZED).dist(new NormalDistribution(0.01, 0.01))
                .seed(seed)
                .constrainGradientToUnitNorm(true)
                .iterations(iterations)
                .learningRate(0.03)
                .momentum(0.5)
                .dropOut(0.5)
                .activationFunction("sigmoid")
                .momentumAfter(Collections.singletonMap(3, 0.9))
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .visibleUnit(RBM.VisibleUnit.BINARY)
                        .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                        .list(4)
                        .layer(0, new RBM.Builder().nIn(numInput).nOut(500).build())
                        .layer(1, new RBM.Builder().nIn(500).nOut(250).build())
                        .layer(2, new RBM.Builder().nIn(250).nOut(200).build())
                        .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.RMSE_XENT).activation("softmax")
                                .nIn(200).nOut(numOut).build())
                        .build();

        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(multiLayerConfiguration);
        multiLayerNetwork.init();

        multiLayerNetwork.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(1)));

        log.info("Splitting dataset - This may take awhile, depending on the input");
        DataSet allData = new DataSet();
        // Loading entirety of dataset into memory
        while(dataSetIterator.hasNext()) {
            allData = dataSetIterator.next();
        }

        SplitTestAndTrain splitTestAndTrain = allData.splitTestAndTrain(0.9);
        DataSet trainingData = splitTestAndTrain.getTrain();

        log.info("Training Network");

        List<DataSet> trainingList = trainingData.asList();
        for(int i = 0 ; i < numEpochs ; i++) {
            Collections.shuffle(trainingList);
            DataSetIterator trainingListIterator = new ListDataSetIterator(trainingList, miniBatchSize);
            multiLayerNetwork.fit(trainingListIterator);
        }

        log.info("Evalauting Network");

        DataSet testingData = splitTestAndTrain.getTest();
        INDArray predict = multiLayerNetwork.output(testingData.getFeatureMatrix());

        evaluation.eval(testingData.getLabels(), predict);

        log.info(evaluation.stats());
    }
}
