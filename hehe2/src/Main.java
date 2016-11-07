import com.sun.deploy.util.StringUtils;

import java.io.*;
import java.util.Arrays;
import java.util.Scanner;

/**
 * Created by jiayao on 2016/11/1.
 */
public class Main {
    public static void main(String[] args) throws IOException {
//        txt2cvs(false);
        String path = "D:\\design(2)\\design\\data\\8、credit\\credit.txt";
        ann2svm(path);
    }

    private static void ann2svm(String path) throws IOException {
        Scanner s = new Scanner(new File(path));
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path.substring(0, path.indexOf(".txt")) + "2svm.txt")));
        while (s.hasNext()) {
            boolean thisLineOk = true;
            String currrent = s.nextLine();
            String[] data = currrent.split(" ");
            String out = "";
            if (data[data.length - 1].equals("0")) out += "-1 ";
            else out += "1 ";
            for (int i = 0; i < data.length - 1; ) {
                if (data[i].equals("?")) {
                    thisLineOk = false;
                    break;
                }
                out += ++i;
                out += ":";
                out += data[i - 1];
                out += " ";
            }
            if (thisLineOk)
                bw.write(out + "\n");
        }
        bw.flush();
        bw.close();
    }


    private static void txt2cvs(boolean ok) throws IOException {
        String now, after;
        if (ok) {
            now = new String(" ");
            after = new String(",");
        } else {
            now = new String(",");
            after = new String(" ");
        }
        String path = "D:\\design(2)\\design\\data\\7、ecoli\\ecoli.txt";
        Scanner s = new Scanner(new File(path));
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path.substring(0, path.indexOf(".txt")) + "_ok.txt")));
        while (s.hasNext()) {
            String str = s.nextLine();
            String[] data = str.split(now);
            bw.write(join(data, after) + "\n");
        }
        bw.flush();
        bw.close();
    }

    private static String join(String[] data, String add) {
        StringBuffer ans = new StringBuffer();
        int i = 0;
        for (; i < data.length - 1; i++) {
            if (data[i].length() == 0) continue;
            ans.append(data[i]).append(add);
        }
        ans.append(data[i]);
        return ans.toString();
    }
}
