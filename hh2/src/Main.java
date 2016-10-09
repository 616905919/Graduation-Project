import java.io.*;
import java.util.*;

/**
 * Created by ImJosMR on 2016/6/15.
 */

/**
 * <li> y is true laber</li>
 * <li> f is feature vector</li>
 * <li> n is the number of features
 */
class Instance {
    public float y;
    public float[] f;
    public int n;

    public Instance() {
    }

    public Instance(float[] floats, int i, float floa) {
        y = floa;
        f = floats;
        n = i;
    }
}

class Instances {
    public ArrayList<Instance> ins;
    public int n;//number of instance ins.length

    public Instances() {
        ins = new ArrayList<Instance>();
//        Collections.shuffle(ins);
        n = 0;
    }
}

class Model implements Cloneable {
    int HiddenNodeNum;
    float[][] HiddenLayer;
    float[] OutputLayer;
    float[][] OutputOfHiddenLayer;

    /**
     * @param n is the number of feature of instance
     * @param i is the number of hidden nodes
     * @param m is the number of instances
     */
    public Model(int n, int i, int m) {
        HiddenLayer = new float[i][n + 1];
        OutputLayer = new float[i + 1];
        OutputOfHiddenLayer = new float[m][i];
        HiddenNodeNum = i;
    }

    @Override
    protected Model clone() throws CloneNotSupportedException {
        //return super.clone();
        Model m = (Model) super.clone();
        return m;
    }
}

/**
 * <li>index is the index of instances </li>
 * <li> z  is the prediction label</li>
 * <li>y is the true label</li>
 */
class pair implements Comparable<pair> {
    int index;
    float z;
    float y;

    public pair(int a, float b, float c) {
        a = index;
        b = z;
        c = y;
    }

    @Override
    public int compareTo(pair o) {
        return new Float(Math.abs(this.z - this.y)).compareTo(Math.abs(o.z - o.y));
    }
}

public class Main {
    public static final float Yita = 0.3f;


    public static void main(String[] args) throws IOException, CloneNotSupportedException {
        File f = new File("C:\\Users\\jiang_000\\Desktop\\5.txt");
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(f)));
        Instances instances = ReadInstance(br);
        Model model = Train(instances, 10, 10000);
    }

    /**
     * @param instances
     * @param i              the number of hidden node
     * @param IterationTimes the max times of iterations
     * @return
     */
    private static Model Train(Instances instances, int i, int IterationTimes) throws CloneNotSupportedException {
        Model model = new Model(instances.ins.get(0).n, i, instances.n);
        //int batch = (int) Math.sqrt(instances.n);
        int batch = instances.n;
        InitModel(model);
        float[] z;//= new float[instances.n];
        float[] y = new float[instances.n];
        for (int j = 0; j < instances.n; j++) {
            y[j] = instances.ins.get(j).y;
        }
        float y2 = InnerProduct(y, y);
        z = ClassifyAll(model, instances);
        while (IterationTimes-- > 0) {
            Collections.shuffle(instances.ins);
            for (int j = 0; j < instances.n; j++) {
                y[j] = instances.ins.get(j).y;
            }
            z = ClassifyAll(model, instances);
            float yz = InnerProduct(y, z);
            float Now_f = 2 * yz / (InnerProduct(z, z) + InnerProduct(y, y));
            if (IterationTimes % 100 == 0)
                System.out.println("int the" + (1000 - IterationTimes) + "times iteration , the approximate f is " + Now_f);
            float z2 = InnerProduct(z, z);
            float y2z2 = y2 + z2;
            for (int j = 0; j < instances.n; j++) {
                Instance instance = instances.ins.get(j);
                float[] tmp = model.OutputLayer;
                float dt = z[j] * (1 - z[j]) * (y[j] / y2z2 - 2 * yz / y2z2 / y2z2 * z[j]);
                int k = 0;
                int l;
                for (; k < model.HiddenNodeNum; k++) {
                    model.OutputLayer[k] += Yita * dt * model.OutputOfHiddenLayer[j][k];
                    l = 0;
                    for (; l < instance.n; l++) {
                        model.HiddenLayer[k][l] += Yita * dt * tmp[k] * model.OutputOfHiddenLayer[j][k] * (1 - model.OutputOfHiddenLayer[j][k]) * instance.f[l];
                    }
                    model.HiddenLayer[k][l] += Yita * dt * model.OutputOfHiddenLayer[j][k] * (1 - model.OutputOfHiddenLayer[j][k]);

                }
                model.OutputLayer[k] += Yita * dt;
            }
        }
        return model;
    }

    private static float[] ClassifyAll(Model model, Instances instances) {
        float[] z = new float[instances.n];
        for (int i = 0; i < z.length; i++) {
            z[i] = ClassifyOne(i, instances.ins.get(i), model);
        }
        return z;
    }

    private static float ClassifyOne(int index, Instance instance, Model model) {
        float[] h = new float[model.HiddenNodeNum];
        for (int i = 0; i < model.HiddenNodeNum; i++) {
            h[i] = Sigmod(InnerProduct(instance.f, model.HiddenLayer[i]));
            model.OutputOfHiddenLayer[index][i] = h[i];
        }
        return Sigmod(InnerProduct(h, model.OutputLayer));
    }

    /**
     * inner product
     *
     * @param f      the first vector
     * @param floats the second vector
     * @return
     */
    private static float InnerProduct(float[] f, float[] floats) {
        float re = 0;
        int i = 0;
        for (; i < f.length; i++) {
            re += f[i] * floats[i];
        }
        if (f.length == floats.length) return re;
        else
            return re + floats[i];
    }

    private static void InitModel(Model model) {
        Random r = new Random();
        for (int i = 0; i < model.HiddenLayer.length; i++) {
            for (int j = 0; j < model.HiddenLayer[0].length; j++) {
                model.HiddenLayer[i][j] = (r.nextFloat() * 0.1f - 0.05f);
            }
        }
        for (int i = 0; i < model.HiddenLayer.length + 1; i++) {
            model.OutputLayer[i] = (r.nextFloat() * 0.1f - 0.05f);
        }
    }

    private static float Sigmod(float f) {
        float ff = (float) (1. / (1 + Math.exp(-f)));
        if (ff < 0.1) return 0.1f;
        else if (ff > 0.9) return 0.9f;
        else
        return ff;
    }

    private static Instances ReadInstance(BufferedReader br) throws IOException {
        String str;
        Instances instances = new Instances();
        int n = -1;
        Scanner s = new Scanner("");
        while ((str = br.readLine()) != null) {
            s = new Scanner(str);
            if (n == -1) {
                List<Float> l = new ArrayList<Float>();
                while (s.hasNext()) {
                    l.add(s.nextFloat());
                }
                float y = l.get(l.size() - 1);
                l.remove(l.size() - 1);
                n = l.size();
                float[] fl = new float[n];
                for (int i = 0; i < fl.length; i++) {
                    fl[i] = l.get(i).floatValue();
                }
                instances.ins.add(new Instance(fl, n, y));
            } else {
                int i = 0;
                float[] fl = new float[n];
                while (i < n) {
                    fl[i++] = s.nextFloat();
                }
                float y = s.nextFloat();
                instances.ins.add(new Instance(fl, n, y));
            }
        }
        instances.n = instances.ins.size();
        return instances;
    }
}

