package com.yaowen.myJava.DataStructure;

public class SegmentTreeNode {
    public int start;
    public int end;
    public int val;
    public SegmentTreeNode left, right;

    public SegmentTreeNode() {
    }

    public SegmentTreeNode(int start, int end, int val) {
        this.start = start;
        this.end = end;
        this.val = val;
        this.left = null;
        this.right = null;
    }

    public SegmentTreeNode(int start, int end, int val, SegmentTreeNode left, SegmentTreeNode right) {
        this.start = start;
        this.end = end;
        this.val = val;
        this.left = left;
        this.right = right;
    }
}
