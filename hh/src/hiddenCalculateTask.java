import com.sun.org.apache.xpath.internal.operations.Bool;

/**
 * Created by jiayao on 2016/11/16.
 */
class HiddenCalculateTask extends Thread {
    public static Boolean[] synSignal;
    public static final Object synLock = new Object();
    public static Boolean syn_all;
    Model model;
    float dt;
    int hiddenNodeNum;
    float[] tmp;
    Instance instance;
    int instanceIndex;
    double Yita;

    public HiddenCalculateTask(Model model, float dt, int k, float[] tmp, Instance instance, int index,
                               double yita) {
        this.model = model;
        this.dt = dt;
        this.hiddenNodeNum = k;
        this.tmp = tmp;
        this.instance = instance;
        this.instanceIndex = index;
        this.Yita = yita;
    }

    @Override
    public void run() {
        int k = hiddenNodeNum;
        int index = instanceIndex;
        model.OutputLayer[k] += Yita * dt * model.OutputOfHiddenLayer[index][k];
        int l = 0;
        float dtHidden = dt * tmp[k] * model.OutputOfHiddenLayer[index][k] * (1 - model.OutputOfHiddenLayer[index][k]);
        for (; l < instance.n; l++) {
            model.HiddenLayer[k][l] += Yita * dtHidden * instance.f[l];
        }
        model.HiddenLayer[k][l] += Yita * dtHidden;
        synchronized (synLock) {
            synSignal[k] = new Boolean(true);
            boolean b = true;
            for (Boolean bo : synSignal) {
                b = b && bo;
            }
            syn_all = b;
            synLock.notifyAll();
        }
    }
}