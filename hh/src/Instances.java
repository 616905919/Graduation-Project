import java.util.ArrayList;

/**
 * Created by jiayao on 2016/11/16.
 */
public class Instances {
    public ArrayList<Instance> ins;
    public int n;//number of instance ins.length
    public int featureNum;

    public Instances() {
        ins = new ArrayList<Instance>();
//        Collections.shuffle(ins);
        n = 0;
        featureNum = 0;
    }
}
