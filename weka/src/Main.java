/**
 * Created by jiayao on 2016/11/7.
 */
public class Main {
    public static void main(String[] args) throws Exception {
//        String data = "D:\\design(2)\\design\\data\\4、beast\\beast.csv.arff";
//        ArffLoader arffLoader = new ArffLoader();
//        arffLoader.setFile(new File(data));
//        Instances instances = arffLoader.getDataSet();
//        instances.setClassIndex(instances.numAttributes() - 1);
//        MultilayerPerceptron multilayerPerceptron = new MultilayerPerceptron();
//        multilayerPerceptron.setNominalToBinaryFilter(false);
//        multilayerPerceptron.buildClassifier(instances);
//        multilayerPerceptron.classifyInstance(instances.get(0));
        new Main().hammingWeight((int) 2147483648L);
    }

//}

    //public class Solution {
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        int ans = 0;
        while (n != 0) {
            ans += n & 0x1;
            n = n >>> 1;
        }
        return ans;
    }
}
