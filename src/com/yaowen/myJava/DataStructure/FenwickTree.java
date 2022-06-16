package com.yaowen.myJava.DataStructure;

public class FenwickTree {
    private final int[] nodes;

    public FenwickTree(int[] array) {
        nodes = new int[array.length + 1];
        for (int i = 0; i < array.length; i++) {
            update(i + 1, array[i]);
        }
    }

    public void update(int index, int val) {
        while (index < nodes.length) {
            nodes[index] += val;
            index += lowBit(index);
        }
    }

    public int getRange(int begin, int end) {
        if (begin > end || end > nodes.length)
            return Integer.MIN_VALUE;

        begin++;
        end++;

        if (begin == 1)
            return prefixSum(end);

        return prefixSum(end) - prefixSum(begin - 1);
    }

    public int[] getNodes() {
        return this.nodes;
    }

    private int prefixSum(int index) {
        int sum = 0;
        while (index > 0) {
            sum += nodes[index];
            index -= lowBit(index);
        }
        return sum;
    }

    private int lowBit(int n) {
        return n & -n;
    }
}
