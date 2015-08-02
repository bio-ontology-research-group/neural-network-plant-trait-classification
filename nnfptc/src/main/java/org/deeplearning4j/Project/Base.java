package org.deeplearning4j.Project;

import org.canova.image.recordreader.ImageRecordReader;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.LoggerFactory;

/**
 * Created by keiron on 02/08/15.
 */
public class Base {
    public static final Logger log = LoggerFactory.getLogger(Base.class);


    final int numInputRows = 200;
    final int numInputCols = 200;

    log.info("Loading labels");
    String labeledPath = System.getProperty("user.home");
    List<String> labels = new ArrayList<>();
    for File f : new File(labeledPath).listFiles()) {
        labels.add(f.getName());
    }

    log.info("Loading data");
    RecordReader imageData = new ImageRecordReader(numInputRows, numInputCols, true);
    imageData.initialize(new FileSplit(new File(labeledPath)));
    DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(imageData, totalImages, numInputRows * numInputCols, labels.size());
    Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

}
