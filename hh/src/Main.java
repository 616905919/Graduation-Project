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
    public int featureNum;

    public Instances() {
        ins = new ArrayList<Instance>();
//        Collections.shuffle(ins);
        n = 0;
        featureNum = 0;
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
    public static final float Yita = 0.1f;
    public static long now = 0;

    public static void main(String[] args) throws IOException, CloneNotSupportedException {
//        File f = new File("D:\\design(2)\\design\\data\\6、haberman\\haberman_ok.txt");
        File f = new File("E:\\git\\Graduation-Project\\data\\1、yeast\\yeast.txt");//文件路径
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(f)));
        Instances instances = ReadInstance(br);
        Collections.shuffle(instances.ins);
        pretreatData(instances);
        now = System.currentTimeMillis();
//        Model fnnmo = TrainFnn(instances, 7, 10000);
        Model model = Train(instances, 6, 100000);
//        TrainKCross(instances, 6, 100000);
        for (int i = 0; i < 1; i++) {
            for (int j = 0; j < 10; j++) {
//                new TrainModel(instances, j, 10000).start();
//                Model model = Train(instances, j, 10000);
            }
        }
        System.out.println("this algorithm cost " + (System.currentTimeMillis() - now) + "ms to train and classify data");
    }

    private static void pretreatData(Instances instances) {
        int featureNum = instances.featureNum;
        float[] ranges = new float[featureNum];
        float[] bases = new float[featureNum];
        for (int i = 0; i < instances.featureNum; i++) {
            float min = Float.POSITIVE_INFINITY;
            float max = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < instances.n; j++) {
                min = Float.min(min, instances.ins.get(j).f[i]);
                max = Float.max(max, instances.ins.get(j).f[i]);
            }
            ranges[i] = (max - min) / 2;
            bases[i] = (max + min) / 2;
        }
        for (int i = 0; i < instances.n; i++) {
            for (int j = 0; j < instances.featureNum; j++) {
                instances.ins.get(i).f[j] = (instances.ins.get(i).f[j] - bases[j]) / ranges[j];
            }
        }
    }

    public static Model TrainFnn(Instances instances, int i, int IterationTimes) throws CloneNotSupportedException {//算法1最大化F1值分类过程的神经网络
        double Yita = Main.Yita / Math.sqrt(instances.n);
        int timenow = 1;
        double maxF = Double.NEGATIVE_INFINITY;
        float Now_f = 0;
        Model model = new Model(instances.ins.get(0).n, i, instances.n);
        Model lastModel = model.clone();
        InitModel(model);
        float[] z;
        float[] y = new float[instances.n];
        for (int j = 0; j < instances.n; j++) {
            y[j] = instances.ins.get(j).y;
        }
        float y2 = InnerProduct(y, y);
        while (timenow++ < IterationTimes) {
            z = ClassifyAll(model, instances, true);
            int[] tmpz = null;
            Now_f = 2 * InnerProduct(z, y) / (y2 + InnerProduct(z, z));
            if (Now_f > maxF) {
                lastModel = model.clone();
            }
            if (IterationTimes - timenow < 10) {
                tmpz = GetMaxExpF(z);
            }
            if (timenow % 100 == 0 || IterationTimes - timenow < 10)
                System.out.println("in the" + timenow + "times iteration , the approximate f is " + Now_f);
            for (int j = 0; j < instances.n; j++) {
                if (IterationTimes - timenow < 10)
                    if (tmpz[j] == y[j]) continue;
                Instance instance = instances.ins.get(j);
                float[] tmp = model.OutputLayer;
                float dt = z[j] * (1 - z[j]) * (y[j] - z[j]);
                int k = 0;
                int l;
                for (; k < model.HiddenNodeNum; k++) {
                    model.OutputLayer[k] += (Yita) * dt * model.OutputOfHiddenLayer[j][k];
                    l = 0;
                    float dtHidden = dt * tmp[k] * model.OutputOfHiddenLayer[j][k] * (1 - model.OutputOfHiddenLayer[j][k]);
                    for (; l < instance.n; l++) {
                        model.HiddenLayer[k][l] += Yita * dtHidden * instance.f[l];
                    }
                    model.HiddenLayer[k][l] += Yita * dtHidden;
                }
                model.OutputLayer[k] += Yita * dt;
            }
        }
        z = ClassifyAll(model, instances, true);
        int zn[] = GetMaxExpF(z);
        int a = 0, b = 0, c = 0, d = 0;
        for (int j = 0; j < zn.length; j++) {
            if (y[j] == 1 && zn[j] == 1) d++;
            else if (y[j] == 1 && zn[j] == 0) c++;
            else if (y[j] == 0 && zn[j] == 0) a++;
            else b++;
        }
        System.out.println("the number of hidden node is " + i + "\n the final f is " + 2 * InnerProduct(z, y) / (y2 + InnerProduct(z, z)));
        System.out.println(a + "\t" + b);
        System.out.println(c + "\t" + d);
        return model;
    }

    public static void TrainKCross(Instances instances, int i, int IterationTimes) throws CloneNotSupportedException {//算法2的交叉验证
        float[] y = new float[instances.n];
        for (int j = 0; j < instances.n; j++) {
            y[j] = instances.ins.get(j).y;
        }
        Instances[] Kinstances = new Instances[3];
        for (int j = 0; j < 3; j++) {
            Kinstances[j] = new Instances();
            if (j == 0)
                Kinstances[j].ins.addAll(instances.ins.subList(instances.n / 3 + 1, instances.n));
            else if (j == 1) {
                Kinstances[j].ins.addAll(instances.ins.subList(0, instances.n / 3 + 1));
                Kinstances[j].ins.addAll(instances.ins.subList(instances.n * 2 / 3 + 1, instances.n));
            } else {
                Kinstances[j].ins.addAll(instances.ins.subList(0, instances.n * 2 / 3 + 1));
            }
            Kinstances[j].n = Kinstances[j].ins.size();
            Kinstances[j].featureNum = instances.featureNum;
        }
        Model model0 = Train(Kinstances[0], i, IterationTimes);
        Model model1 = Train(Kinstances[1], i, IterationTimes);
        Model model2 = Train(Kinstances[2], i, IterationTimes);
        float[][] ans = new float[3][];
        ans[0] = ClassifyAll(model0, instances);
        ans[1] = ClassifyAll(model1, instances);
        ans[2] = ClassifyAll(model2, instances);
        float[] trueAns = new float[instances.n];
        System.arraycopy(ans[0], 0, trueAns, 0, instances.n - Kinstances[0].n);
        System.arraycopy(ans[1], instances.n / 3 + 1, trueAns, instances.n / 3 + 1, instances.n - Kinstances[1].n);
        System.arraycopy(ans[2], instances.n * 2 / 3 + 1, trueAns, instances.n * 2 / 3 + 1, instances.n - Kinstances[2].n);
        printPerformance(trueAns, y, i);
    }

    /**
     * @param instances
     * @param i              the number of hidden node
     * @param IterationTimes the max times of iterations
     * @return
     */

    public static Model Train(Instances instances, int i, int IterationTimes) throws CloneNotSupportedException {//基于最小化损失学习的神经网络
        int timeNow = 1;
        double fLast = 0;
        int maxDonotUp = 3;//可以调整 影响最终精度
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
        while (timeNow++ < IterationTimes) {
//            if (System.currentTimeMillis() - now > 300000) break all;
            //Collections.shuffle(instances.ins);
            double Yita = Math.sqrt(Main.Yita * timeNow);
            z = ClassifyAll(model, instances, false);
            float yz = InnerProduct(y, z);
            float Now_f = 2 * yz / (InnerProduct(z, z) + InnerProduct(y, y));
            if (timeNow % 1 == 0) {
                if (timeNow % 500 == 0) {//可以调整 影响最终精度
                    if (Now_f - fLast < 0.001)//可以调整 影响最终精度
                        break all;
                    fLast = Now_f;
                }
                if (timeNow % 20 == 0)
                    System.out.println("in the" + timeNow + "times iteration , the approximate f is " + Now_f);
            }
            float z2 = InnerProduct(z, z);
            float y2z2 = y2 + z2;
            Model LastModel = model.clone();
            int batchNow = batch;
            int j = 0;
            outer:
            while (true) {
                if (j + batchNow > instances.n) {
                    j = 0;
                    batchNow = (int) (batchNow * 0.8);
                    //System.out.println(batchNow);
                    if (batchNow <= instances.n * 0.001) {
                        System.out.println(maxDonotUp);
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
                    //使用线程池以及多线程 可以在隐藏节点数目较多的时候 优化每次迭代速度
                    //↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 用wait notify
                    for (; k < model.HiddenNodeNum; k++) {
                        model.OutputLayer[k] += Yita * dt * model.OutputOfHiddenLayer[index][k];
                        l = 0;
                        float dtHidden = dt * tmp[k] * model.OutputOfHiddenLayer[index][k] * (1 - model.OutputOfHiddenLayer[index][k]);
                        for (; l < instance.n; l++) {
                            model.HiddenLayer[k][l] += Yita * dtHidden * instance.f[l];
                        }
                        model.HiddenLayer[k][l] += Yita * dtHidden;
                    }
                    model.OutputLayer[k] += Yita * dt;
                }
                //↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
                z = ClassifyAll(model, instances, false);
                float f_new = 2 * InnerProduct(y, z) / (InnerProduct(z, z) + InnerProduct(y, y));
                j += batchNow;
                if (f_new > Now_f)
                    break outer;
            }
        }
        z = ClassifyAll(model, instances, true);
//        float[] ztmp = z;
//        for (int j = 0; j < ztmp.length; j++) {
//            if (ztmp[j] < 0.1) ztmp[j] = 0;
//            else if (ztmp[j] > 0.9) ztmp[j] = 1;
//        }
//        int[] z2 = GetMaxExpF(ztmp);
//        System.out.println("the number of hidden node is " + i + "\nthe f of max expectation algorithm is" + 2 * InnerProduct(z2, y) / (y2 + InnerProduct(z2)));
        printPerformance(z, y, i);
//        int a = 0, b = 0, c = 0, d = 0;
//        for (int j = 0; j < z.length; j++) {
//            z[j] = z[j] >= 0.5 ? 1 : 0;
//            if (y[j] == 1 && z[j] == 1) d++;
//            else if (y[j] == 1 && z[j] == 0) c++;
//            else if (y[j] == 0 && z[j] == 0) a++;
//            else b++;
//        }
//        System.out.println("the number of hidden node is " + i + "\n the final f is " + 2 * InnerProduct(z, y) / (y2 + InnerProduct(z, z)));
//        System.out.println(a + "\t" + b);
//        System.out.println(c + "\t" + d);
        return model;
    }

    private static void printPerformance(float[] z, float[] y, int i) {
        int a = 0, b = 0, c = 0, d = 0;
        for (int j = 0; j < z.length; j++) {
            z[j] = z[j] >= 0.5 ? 1 : 0;
            if (y[j] == 1 && z[j] == 1) d++;
            else if (y[j] == 1 && z[j] == 0) c++;
            else if (y[j] == 0 && z[j] == 0) a++;
            else b++;
        }
        System.out.println("the number of hidden node is " + i + "\n the final f is " + 2 * InnerProduct(z, y) / (InnerProduct(y, y) + InnerProduct(z, z)));
        System.out.println(a + "\t" + b);
        System.out.println(c + "\t" + d);
        new Exception().printStackTrace();
    }
    //2.3 里面那个算法的实现 以及优化
    private static int[] GetMaxExpF(float[] z) {
        long time = System.currentTimeMillis();
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
//        System.out.println("e is " + e);
//        System.out.println("run  get max f in " + (System.currentTimeMillis() - time) + "ms");
        return ans;
    }

    /**
     * @param z2 is the probability of evert instance that predicted to be positive
     * @param z  is the zk of predicted
     * @param zp is k
     * @return is the expectation of f when k is zp and zk is z
     */
    private static float getExpf(int[] z2, float[] z, int zp) {
        float[] ztmp = Arrays.copyOf(z, z.length);
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
                if (ztmp[i] >= 0.89) ztmp[i] = 1;
                if (ztmp[i] <= 0.11) ztmp[i] = 0;
                if (N.containsKey(s0)) {
                    N.put(s0, N.get(s0) + p * (1 - ztmp[i]));
                } else N.put(s0, p * (1 - ztmp[i]));
                if (N.containsKey(s1)) {
                    N.put(s1, N.get(s1) + p * ztmp[i]);
                } else N.put(s1, p * ztmp[i]);
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
            if (s.t + zp == 0) ans += 0;
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

    private static float[] ClassifyAll(Model model, Instances instances) {
        float[] z = new float[instances.n];
        for (int i = 0; i < z.length; i++) {
            z[i] = ClassifyOne(instances.ins.get(i), model);
        }
        return z;
    }

    private static float ClassifyOne(Instance instance, Model model) {
        float[] h = new float[model.HiddenNodeNum];
        for (int i = 0; i < model.HiddenNodeNum; i++) {
            h[i] = Sigmod(InnerProduct(instance.f, model.HiddenLayer[i]), true);
        }
        return Sigmod(InnerProduct(h, model.OutputLayer), true);
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
//                model.HiddenLayer[i][j] = (r.nextFloat() * 2f - 1f);
            }
        }
        for (int i = 0; i < model.HiddenLayer.length + 1; i++) {
            model.OutputLayer[i] = (r.nextFloat() * 0.1f - 0.05f);
//            model.OutputLayer[i] = (r.nextFloat() * 2f - 1f);
        }
    }

    private static float Sigmod(float f, boolean classify) {
        float ff = (float) (1. / (1 + Math.exp(-f)));
        return ff;
//        if (!classify) {
//            if (ff < 0.1) return 0.1f;
//            else if (ff > 0.9) return 0.9f;
//            else
//                return ff;
//        } else return ff;
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
                float y = l.remove(l.size() - 1);
                n = l.size();
                float[] fl = new float[n];
                for (int i = 0; i < fl.length; i++) {
                    fl[i] = l.get(i).floatValue();
                }
                instances.ins.add(new Instance(fl, n, y));
            } else {
                int i = 0;
                float[] fl = new float[n];
                try {
                    while (i < n) {
                        fl[i++] = s.nextFloat();
                    }
                    float y = s.nextFloat();
                    instances.ins.add(new Instance(fl, n, y));
                } catch (Exception e) {
//                    System.out.println(str);
//                    e.printStackTrace();
                }

            }
        }
        instances.n = instances.ins.size();
        instances.featureNum = n;
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

