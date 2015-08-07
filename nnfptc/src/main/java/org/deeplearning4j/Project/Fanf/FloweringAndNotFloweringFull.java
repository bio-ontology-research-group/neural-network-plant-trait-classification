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
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.plot.iterationlistener.GradientPlotterIterationListener;
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
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Created by osheak on 03/08/15.
 */
public class FloweringAndNotFloweringFull {
    public static final Logger log = LoggerFactory.getLogger(FloweringAndNotFloweringFull.class);

    public static void main (String args[]) throws Exception {
        String labeledPath = System.getProperty("user.home")+"/datasets/fanfMid/";
        int numNF = 1000;
        int numF = 1000;
        int numInputRows = 28;
        int numInputCols = 28;
        int numImages = numF + numNF;
        List<String> labels = new ArrayList<>();

        int numInput = numInputRows * numInputCols;
        int iterations = 5;
        int seed = 123;

        final int numEpochs = 20;
        final int miniBatchSize = 100;
        Evaluation evaluation = new Evaluation();

        log.info("Loading labels");
        for(File f : new File(labeledPath).listFiles()) {
            labels.add(f.getName());
        }

        int numOut = labels.size();


        log.info("Loading data");

        RecordReader recordReader = new ImageRecordReader(numInputRows, numInputCols, true, labels);
        recordReader.initialize(new FileSplit(new File(labeledPath)));
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, numImages, numInput, labels.size());
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        log.info("Building model");


        MultiLayerConfiguration multiLayerConfiguration = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .seed(seed)
                .iterations(iterations)
                .maxNumLineSearchIterations(10) // Magical Optimisation Stuff
                .activationFunction("relu")
                .k(1) // Annoying dl4j bug that is yet to be fixed.
                .weightInit(WeightInit.XAVIER)
                .constrainGradientToUnitNorm(true)
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .regularization(true)
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .list(4)
                .layer(0, new RBM.Builder().nIn(numInput).nOut(500).build())
                .layer(1, new RBM.Builder().nIn(500).nOut(250).build())
                .layer(2, new RBM.Builder().nIn(250).nOut(100).build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation("softmax")
                        .nIn(100).nOut(numOut).build())
                        // Pretrain is unsupervised pretraining and finetuning on output layer
                        // Backward is full propagation on ALL layers.
                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(multiLayerConfiguration);
        multiLayerNetwork.init();

        multiLayerNetwork.setListeners(Arrays.<IterationListener>asList(new ScoreIterationListener(1), new GradientPlotterIterationListener(5)));

        log.info("Splitting dataset - This may take awhile, depending on the input");

        DataSet allData = new DataSet();
        // Loading Entirety of dataSetIterator into memory.
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

        log.info("Evaluating Network");

        DataSet testingData = splitTestAndTrain.getTest();
        INDArray predict = multiLayerNetwork.output(testingData.getFeatureMatrix());

        evaluation.eval(testingData.getLabels(), predict);

        log.info(evaluation.stats());
    }
}
