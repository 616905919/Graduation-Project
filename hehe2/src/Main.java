import com.sun.deploy.util.StringUtils;

import java.io.*;
import java.util.Arrays;
import java.util.Scanner;

/**
 * Created by jiayao on 2016/11/1.
 */
public class Main {
    public static void main(String[] args) throws IOException {
        txt2cvs(false);
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
        Scanner s = new Scanner(new File("C:\\Users\\jiayao\\Desktop\\credit.txt"));
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("C:\\Users\\jiayao\\Desktop\\credit.txt")));
        while (s.hasNext()) {
            String str = s.nextLine();
            String[] data = str.split(now);
            bw.write(join(data, after) + "\n");
        }
        bw.flush();
        bw.close();
    }
    private static String join(String[] data , String add){
        StringBuffer ans = new StringBuffer();
        int i = 0;
        for (;i<data.length-1;i++){
            if(data[i].length()==0) continue;
            ans.append(data[i]).append(add);
        }
        ans.append(data[i]);
        return ans.toString();
    }
}
