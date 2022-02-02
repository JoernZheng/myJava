package com.yaowen.myJava;

import java.util.*;

public class Main {
    public static void main(String[] args) {
        int[][] twoDim = { {1, 2}, {3, 7}, {8, 9}, {4, 2}, {5, 3} };

        Arrays.sort(twoDim, (a1, a2) -> a2[0] - a1[0]);

        System.out.println(Arrays.deepToString(twoDim));

    }
}
