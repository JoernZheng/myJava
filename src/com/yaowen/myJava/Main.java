package com.yaowen.myJava;

import com.yaowen.myJava.DataStructure.SegmentTree;

public class Main {
    public static void main(String[] args) {
        int[] values = new int[]{1,3,8,13,15,17};
        SegmentTree segmentTree = new SegmentTree(values);
        segmentTree.printTree();
        System.out.println(segmentTree.rangeQuery(2, 4));
        System.out.println(segmentTree.rangeQuery(1, 1));
        System.out.println(segmentTree.rangeQuery(3, 5));
        segmentTree.update(2, 9);
        segmentTree.printTree();
    }
}
