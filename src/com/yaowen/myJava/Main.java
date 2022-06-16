package com.yaowen.myJava;

import com.yaowen.myJava.DataStructure.FenwickTree;
import com.yaowen.myJava.DataStructure.SegmentTree;

import java.util.*;
import java.util.stream.Collectors;

public class Main {
    public static void main(String[] args) {
        FenwickTreeTest();
    }

    public static void segmentTreeTest() {
        int[] values = new int[]{1, 3, 8, 13, 15, 17};
        SegmentTree segmentTree = new SegmentTree(values);
        segmentTree.printTree();
        System.out.println(segmentTree.rangeQuery(2, 4));
        System.out.println(segmentTree.rangeQuery(1, 1));
        System.out.println(segmentTree.rangeQuery(3, 5));
        segmentTree.update(2, 9);
        segmentTree.printTree();
    }

    public static void DiagonalTraverseTest() {
        Solution solution = new Solution();
        int[][] testCase = new int[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        System.out.println(Arrays.toString(solution.findDiagonalOrder(testCase)));
    }

    public static void DequeStreamTest() {
        Deque<Integer> deque = new LinkedList<>();
        deque.add(1);
        deque.add(2);
        System.out.println(deque);
        List<Integer> list = deque.stream().map(integer -> {
            integer = integer + 1;
            integer = integer + 2;
            return integer;
        }).filter(integer -> integer > 4).collect(Collectors.toList());
        System.out.println(list);
    }

    public static void FenwickTreeTest() {
        int[] array = new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        FenwickTree fenwickTree = new FenwickTree(array);
        System.out.println(Arrays.toString(fenwickTree.getNodes()));
        System.out.println(fenwickTree.getRange(0, 0));
        System.out.println(fenwickTree.getRange(3, 8));
        System.out.println(fenwickTree.getRange(0, 14));
    }
}
