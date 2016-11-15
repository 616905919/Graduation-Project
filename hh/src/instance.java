/**
 * Created by jiayao on 2016/11/16.
 */

/**
 * <li> y is true laber</li>
 * <li> f is feature vector</li>
 * <li> n is the number of features
 */
public class Instance {
    public float y;
    public float[] f;
    public int n;

    public Instance() {
    }

    public Instance(float[] feature, int i, float label) {
        y = label;
        f = feature;
        n = i;
    }
}