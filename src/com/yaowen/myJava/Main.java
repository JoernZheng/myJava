package com.yaowen.myJava;

import java.util.*;

public class Main {
    public static void main(String[] args) {
        // x(t) = (R+r)*cos((r/R)*t) - a*cos((1+r/R)*t)
        // y(t) = (R+r)*sin((r/R)*t) - a*sin((1+r/R)*t)
        // R=8, r=1, a=4
        double R = 8, r = 1, a = 4, n = 16, step = 0.01, PI = Math.PI;
        double lon = -118.289153, lat = 34.021331;
        List<double[]> list = new LinkedList<>();
        for (double t = 0; t < n * PI; t += step) {
            double x = ((R + r) * Math.cos((r / R) * t) - a * Math.cos((1 + r / R) * t)) / 10000.0;
            double y = ((R + r) * Math.sin((r / R) * t) - a * Math.sin((1 + r / R) * t)) / 10000.0;
            list.add(new double[]{x + lon, y + lat});
            System.out.println((x + lon) + "," + (y + lat));
        }
        System.out.println(list.size());
    }
}
