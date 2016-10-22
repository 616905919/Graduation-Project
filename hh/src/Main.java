import com.sun.xml.internal.ws.util.StringUtils;

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

    public Instance(float[] feature, int i, float label) {
        y = label;
        f = feature;
        n = i;
    }
}

/**
 * <li> a means z*y</li>
 * <li> t means y*y</li>
 * <li> zhe z*z (p)is fixed</li>
 * <li>we use a,p,t to calculate F-measure</li>
 */
class state {
    int a;
    int t;

    public state(int a, int t) {
        this.a = a;
        this.t = t;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null) return false;
        if (getClass() != obj.getClass()) return false;
        state s = (state) obj;
        return s.hashCode() == this.hashCode();
    }

    @Override
    public int hashCode() {
        return a * 10000 + t;
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

class TrainModel extends Thread {
    Instances instances;
    int i;
    int IterationTimes;

    public TrainModel(Instances instances, int a, int b) {
        this.instances = instances;
        this.i = a;
        this.IterationTimes = b;
    }

    @Override
    public void run() {
        try {
//            synchronized (this) {
            Main.Train(instances, i, IterationTimes);
//            }
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
        }
    }
}

class Zk implements Comparable<Zk> {
    float z;
    int index;

    public Zk(float z, int index) {
        this.z = z;
        this.index = index;
    }


    @Override
    public int compareTo(Zk o) {
        return new Float(this.z).compareTo(new Float(o.z));
    }
}

public class Main {
    public static final float Yita = 0.3f;

    public static void main(String[] args) throws IOException, CloneNotSupportedException {
        float[] f = new float[]{0.9f,0.6f,0.51f,0.1f};
        int[] ans = GetMaxExpF(f);
        for(int  i :ans){
            System.out.println(i);
        }
//        File f = new File("C:\\Users\\jiang_000\\Desktop\\1.txt");
//        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(f)));
//        Instances instances = ReadInstance(br);
//        //Model model = Train(instances, 10, 10000);
//        for (int i = 0; i < 2; i++) {
//            for (int j = 0; j < 10; j++) {
//                new TrainModel(instances, j, 10000).start();
//                //Model model = Train(instances, j, 10000);
//            }
//        }
    }

    /**
     * @param instances
     * @param i              the number of hidden node
     * @param IterationTimes the max times of iterations
     * @return
     */
    public static Model Train(Instances instances, int i, int IterationTimes) throws CloneNotSupportedException {
        int maxDonotUp = 15;
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
        z = ClassifyAll(model, instances, false);
        all:
        while (IterationTimes-- > 0) {
            //Collections.shuffle(instances.ins);
            z = ClassifyAll(model, instances, false);
            for (int j = 0; j < instances.n; j++) {
                y[j] = instances.ins.get(j).y;
            }
            float yz = InnerProduct(y, z);
            float Now_f = 2 * yz / (InnerProduct(z, z) + InnerProduct(y, y));
            //if (IterationTimes % 1 == 0)
            //    System.out.println("int the" + (100000 - IterationTimes) + "times iteration , the approximate f is " + Now_f);
            float z2 = InnerProduct(z, z);
            float y2z2 = y2 + z2;
            Model LastModel = model.clone();
            int batchNow = batch;
            int j = 0;
            outer:
            while (true) {
                if (j + batchNow > instances.n) {
                    j = 0;
                    batchNow = (int) (batchNow * 0.9);
                    //System.out.println(batchNow);
                    if (batchNow == 0) {
                        //System.out.println(maxDonotUp);
                        if (maxDonotUp-- <= 0)
                            break all;
                        else break outer;
                    }
                }
                model = LastModel.clone();
                for (int mm = 0; mm < batchNow; mm++) {
                    int index = j + mm;
                    Instance instance = instances.ins.get(index);
                    float[] tmp = model.OutputLayer;
                    float dt = z[index] * (1 - z[index]) * (y[index] / y2z2 - 2 * yz / y2z2 / y2z2 * z[index]);
                    int k = 0;
                    int l;
                    for (; k < model.HiddenNodeNum; k++) {
                        model.OutputLayer[k] += Yita * dt * model.OutputOfHiddenLayer[index][k];
                        l = 0;
                        for (; l < instance.n; l++) {
                            model.HiddenLayer[k][l] += Yita * dt * tmp[k] * model.OutputOfHiddenLayer[index][k] * (1 - model.OutputOfHiddenLayer[index][k]) * instance.f[l];
                        }
                        model.HiddenLayer[k][l] += Yita * dt * model.OutputOfHiddenLayer[index][k] * (1 - model.OutputOfHiddenLayer[index][k]);
                    }
                    model.OutputLayer[k] += Yita * dt;
                }
                z = ClassifyAll(model, instances, false);
                float f_new = 2 * InnerProduct(y, z) / (InnerProduct(z, z) + InnerProduct(y, y));
                j += batchNow;
                if (f_new > Now_f)
                    break outer;
            }
        }
        z = ClassifyAll(model, instances, true);
        float[] ztmp = z;
        for (int j = 0; j < ztmp.length; j++) {
            if (ztmp[j] < 0.1) ztmp[j] = 0;
            else if (ztmp[j] > 0.9) ztmp[j] = 1;
        }
        int[] z2 = GetMaxExpF(ztmp);
        System.out.println("the number of hidden node is " + i + "\nthe f of max expectation algorithm is" + 2 * InnerProduct(z2, y) / (y2 + InnerProduct(z2)));
        for (int j = 0; j < z.length; j++) {
            z[j] = z[j] >= 0.5 ? 1 : 0;
        }
        System.out.println("the number of hidden node is " + i + "\n the final f is " + 2 * InnerProduct(z, y) / (y2 + InnerProduct(z, z)));
        return model;
    }

    private static int[] GetMaxExpF(float[] z) {
        int[] ans = null;
        float e = -1;
        for (int i = 0; i < z.length; i++) {
            int[] z2 = GetZk(z, i);
            float eNow = getExpf(z2, z, i);
//            System.out.println(i);
            if (eNow > e) {
                e = eNow;
                ans = z2;
            }
        }
        System.out.println(e);
        return ans;
    }

    /**
     * @param z2 is the probability of evert instance that predicted to be positive
     * @param z  is the zk of predicted
     * @param zp is k
     * @return is the expectation of f when k is zp and zk is z
     */
    private static float getExpf(int[] z2, float[] z, int zp) {
        Map<state, Float> M = new HashMap<state, Float>();
        M.put(new state(0, 0), new Float(1));
        for (int i = 0; i < z.length; i++) {
            Iterator it = M.entrySet().iterator();
            Map<state, Float> N = new HashMap<state, Float>();
            while (it.hasNext()) {
                Map.Entry<state, Float> e = (Map.Entry<state, Float>) it.next();
                state s = e.getKey();
                Float p = e.getValue();
                if (p == 0) continue;
                state s0 = GetState(s, z2[i], 0);
                state s1 = GetState(s, z2[i], 1);
                if (N.containsKey(s0)) {
                    N.put(s0, N.get(s0) + p * (1 - z[i]));
                } else N.put(s0, p * (1 - z[i]));
                if (N.containsKey(s1)) {
                    N.put(s1, N.get(s1) + p * z[i]);
                } else N.put(s1, p * z[i]);
            }
            M.clear();
            M.putAll(N);
        }
        float ans = 0;
        Iterator<Map.Entry<state, Float>> it = M.entrySet().iterator();
//        float allp = 0;
        while (it.hasNext()) {
            Map.Entry<state, Float> e = it.next();
            state s = e.getKey();
            Float p = e.getValue();
//            allp += p.floatValue();
            if (s.t + zp == 0) ans += p;
            else ans += p * ((s.a + 0.0) / (s.t + zp));
        }
        return ans;
    }

    private static state GetState(state s, int v, int i) {
        state ans = new state(s.a, s.t);
        if (i == 1) {
            ans.t++;
            if (v == 1) ans.a++;
        }
        return ans;
    }


    private static float[] ClassifyAll(Model model, Instances instances, boolean classify) {
        float[] z = new float[instances.n];
        if (!classify) {
            for (int i = 0; i < z.length; i++) {
                z[i] = ClassifyOne(i, instances.ins.get(i), model, false);
            }
        } else {
            for (int i = 0; i < z.length; i++) {
                z[i] = ClassifyOne(i, instances.ins.get(i), model, true);
            }
        }
        return z;
    }

    private static float ClassifyOne(int index, Instance instance, Model model, boolean classify) {
        float[] h = new float[model.HiddenNodeNum];
        if (!classify) {
            for (int i = 0; i < model.HiddenNodeNum; i++) {
                h[i] = Sigmod(InnerProduct(instance.f, model.HiddenLayer[i]), false);
                model.OutputOfHiddenLayer[index][i] = h[i];
            }
        } else {
            for (int i = 0; i < model.HiddenNodeNum; i++) {
                h[i] = Sigmod(InnerProduct(instance.f, model.HiddenLayer[i]), true);
                model.OutputOfHiddenLayer[index][i] = h[i];
            }
        }
        if (!classify)
            return Sigmod(InnerProduct(h, model.OutputLayer), false);
        else return Sigmod(InnerProduct(h, model.OutputLayer), true);
    }

    /**
     * <em>inner product</em>
     * <li> the second vector can have one more dimension than the first one, if it is ,
     * the return will plus the last dimension of the second vector multiplied one</li>
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

    private static float InnerProduct(int[] i) {
        int ans = 0;
        for (int in : i) {
            if (in == 1) ans++;
        }
        return ans;
    }

    private static float InnerProduct(int[] i, float[] f) {
        float ans = 0;
        for (int j = 0; j < i.length; j++) {
            ans += i[j] * f[j];
        }
        return ans;
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

    private static float Sigmod(float f, boolean classify) {
        float ff = (float) (1. / (1 + Math.exp(-f)));
        if (!classify) {
            if (ff < 0.1) return 0.1f;
            else if (ff > 0.9) return 0.9f;
            else
                return ff;
        } else return ff;
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

    private static int[] GetZk(float[] z, int k) {
        int[] ans = new int[z.length];
        if (k == 0) return ans;
        Queue<Zk> q = new PriorityQueue<>();
        int i = 0;
        for (; i < k; i++) {
            q.add(new Zk(z[i], i));
        }
        for (; i < z.length; i++) {
            if (q.peek().z < z[i]) {
                q.poll();
                q.add(new Zk(z[i], i));
            }
        }
        while (!q.isEmpty()) {
            ans[q.poll().index] = 1;
        }
        return ans;
    }
}

