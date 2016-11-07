import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.gui.beans.DataSource;

import java.io.File;
import java.io.IOException;

/**
 * Created by jiayao on 2016/11/7.
 */
public class Main {
    public static void main(String[] args) throws Exception {
        String data = "D:\\design(2)\\design\\data\\4、beast\\beast.csv.arff";
        ArffLoader arffLoader = new ArffLoader();
        arffLoader.setFile(new File(data));
        Instances instances = arffLoader.getDataSet();
        instances.setClassIndex(instances.numAttributes() - 1);
        MultilayerPerceptron multilayerPerceptron = new MultilayerPerceptron();
        multilayerPerceptron.setNominalToBinaryFilter(false);
        multilayerPerceptron.buildClassifier(instances);
        multilayerPerceptron.classifyInstance(instances.get(0));
    }
}
