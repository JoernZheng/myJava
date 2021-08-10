package com.yaowen.myJava;

import java.math.BigInteger;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Solution {
    // 只出现一次的数字
    public int singleNumber(int[] nums) {
        int result = 0;
        for (int num : nums) {
            result = result ^ num;
        }
        return result;
    }

    public int singleNumber2(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            if (map.containsKey(num)) {
                map.remove(num);
            } else {
                map.put(num, 1);
            }
        }
        Set<Integer> set = map.keySet();
        Iterator<Integer> iterator = set.iterator();

        return iterator.next();
    }

    public int[] twoSum(int[] numbers, int target) {
        // 72ms
        for (int i = 0; i < numbers.length; i++) {
            int resultIndex = Arrays.binarySearch(Arrays.copyOfRange(numbers, i + 1, numbers.length), target - numbers[i]);
            if (resultIndex >= 0) {
                return new int[]{i + 1, resultIndex + i + 2};      // i + 1（新数组的0就是旧数组的i+1） + 1（题目从1开始输出）
            }
        }
        return null;
    }

    public int[] twoSum2(int[] numbers, int target) {
        // two points
        // 1ms 83%
        int a = 0, b = numbers.length - 1;
        while (a != b) {
            if (numbers[a] + numbers[b] == target) {
                break;
            } else if (numbers[a] + numbers[b] > target) {
                b--;
            } else {
                a++;
            }
        }
        return new int[]{a + 1, b + 1};
    }

    public int[] twoSum3(int[] numbers, int target) {
        // 二分结合双指针
        int i = 0, j = numbers.length - 1;
        while (i < j) {
            int m = (i + j) >>> 1;
            if (numbers[i] + numbers[m] > target) {
                j = m - 1;
            } else if (numbers[m] + numbers[j] < target) {
                i = m + 1;
            } else if (numbers[i] + numbers[j] > target) {
                j--;
            } else if (numbers[i] + numbers[j] < target) {
                i++;
            } else {
                return new int[]{i + 1, j + 1};
            }
        }
        return new int[]{0, 0};
    }

    // 相交链表
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        // A和B如果相交，则起点到相交点的距离可能不同，但相交点后的距离相同
        // 所以为了消除不同，让A和B指针都走过“自己和对方的 起点到相交点 的距离即可”
        // A和B所遍历的元素个数为 “A离相交点的距离 + B离相交点的距离 + 相交点到Null的距离”
        ListNode current1 = headA, current2 = headB;
        boolean a = false, b = false;
        while (true) {
            if (current1 == current2) {
                return current1;
            } else {
                if (current1.next == null) {
                    if (a) {
                        break;
                    } else {
                        current1 = headB;
                        a = true;
                    }
                } else {
                    current1 = current1.next;
                }
                if (current2.next == null) {
                    if (b) {
                        break;
                    } else {
                        current2 = headA;
                        b = true;
                    }
                } else {
                    current2 = current2.next;
                }
            }
        }
        return null;
    }

    // Excel表列名称
    public String convertToTitle(int columnNumber) {
        StringBuilder result = new StringBuilder();
        while (columnNumber > 0) {
            columnNumber = columnNumber - 1;    // 因为A从1开始至26，所以在计算初始，将范围转化到0~25就相当于普通的26进制计算和转化
            result.append((char) ('A' + (columnNumber % 26)));
            columnNumber = columnNumber / 26;
        }
        return result.reverse().toString();
    }

    // 多数元素
    public int majorityElement(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            if (map.containsKey(num)) {
                map.put(num, map.get(num) + 1);
            } else {
                map.put(num, 1);
            }
        }
        int max_num = 0, max_count = 0;
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            if (entry.getValue() > max_count) {
                max_num = entry.getKey();
            }
        }
        return max_num;
    }

    // 阶乘后的0-解法1
    public int trailingZeroes1(int n) {
        // 计算“5”的个数
        int count = 0;
        for (int i = 1; i <= n; i++) {
            int temp = i;
            while (temp % 5 == 0 && temp != 0) {
                count++;
                temp = temp / 5;
            }
        }
        return count;
    }

    // 阶乘后的0-解法2
    public int trailingZeroes2(int n) {
        int count = 0;
        BigInteger bigInteger = BigInteger.ONE;
        for (int i = 1; i <= n; i++) {
            bigInteger = bigInteger.multiply(BigInteger.valueOf(i));
        }
        while (bigInteger.mod(BigInteger.TEN).equals(BigInteger.ZERO)) {
            count++;
            bigInteger = bigInteger.divide(BigInteger.TEN);
        }
        return count;
    }

    // 阶乘后的0-解法3 高效地进行因子5的计算
    public int trailingZeroes3(int n) {
        int zeroCount = 0;
        // We need to use long because currentMultiple can potentially become
        // larger than an int.
        long currentMultiple = 5;
        while (n >= currentMultiple) {
            zeroCount += (n / currentMultiple);
            currentMultiple *= 5;
        }
        return zeroCount;
    }

    // 颠倒的二进制位 -> API
    public int reverseBits1(int n) {
        return Integer.reverse(n);
    }

    // 颠倒的二进制位 -> 逻辑右移
    public int reverseBits2(int n) {
        int result = 0;
        for (int i = 0; i < 32; i++) {
            result |= n & 1;
            result = result << 1;
            n = n >>> 1;
        }
        return result;
    }

    // 打开转盘锁
    public int openLock(String[] deadends, String target) {
        int count = 0;
        Queue<String> queue1 = new LinkedList<>();
        Set<String> used = Stream.of(deadends).collect(Collectors.toSet()); // 将死亡数值看作是已经访问的位置
        queue1.offer("0000");
        while (!queue1.isEmpty()) {
            // track1
            // 获取当前层次中节点的个数，否则无法统计在第几层
            int size = queue1.size();
            while (size-- > 0) {
                String curr = queue1.poll();
                if (curr.equals(target)) {
                    return count;
                } else if (used.contains(curr)) {
                    continue;
                } else {
                    used.add(curr);
                    for (int i = 0; i < 4; i++) {
                        char c = curr.charAt(i);
                        String plus = curr.substring(0, i) + (c == '9' ? '0' : (char) (c + 1)) + curr.substring(i + 1);
                        String desc = curr.substring(0, i) + (c == '0' ? '9' : (char) (c - 1)) + curr.substring(i + 1);
                        System.out.println(plus);
                        queue1.offer(plus);
                        queue1.offer(desc);
                    }
                }
            }
            count++;
        }
        return -1;
    }

    // 克隆图
    public Node cloneGraph(Node node) {
        // 使用辅助函数来实现真正的逻辑
        // 这里不能使用Set来实现，因为Set没有便携的查找API
        // 哈希表以“值”（唯一）为主键，以Node节点地址为值，有助于方便后续插入操作
        return myCloneGraph(node, new HashMap<>());
    }

    public Node myCloneGraph(Node node, Map<Integer, Node> visited) {
        // 这种情况实际上只有头节点会出现
        if (node == null) return null;
        // 在遍历节点的过程中，只有两种情况：
        // 1.当前节点已经被访问过，即访问到了环，那么只需要把边连接，不需要进入该节点搜索
        // 2.当前节点还未被访问过，那么需要创建当前节点，然后以当前节点为头节点创建后续的节点，然后返回给上一层，这里有点类似于“树的创建”这类的递归思路。
        if (visited.containsKey(node.val)) {
            return visited.get(node.val);
        } else {
            Node temp = new Node(node.val);
            // 需要将新的节点放入到哈希表中以供后面的搜索和取用
            visited.put(node.val, temp);
            for (Node next : node.neighbors) {
                temp.neighbors.add(myCloneGraph(next, visited));
            }
            return temp;
        }
    }

    // 目标和

    private int count = 0;

    public int findTargetSumWays(int[] nums, int target) {
        findTargetSumWaysDFS(nums, 0, nums.length - 1, target);
        return count;
    }

    public void findTargetSumWaysDFS(int[] nums, int start, int end, int target) {
        // 有效路径
        if (start == end) {
            if (Math.abs(nums[end]) == Math.abs(target)) {
                count++;
                if (target == 0) count++;
            }
            return;
        }
        findTargetSumWaysDFS(nums, start + 1, end, target - nums[start]);
        findTargetSumWaysDFS(nums, start + 1, end, target + nums[start]);
    }

    public static class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
            next = null;
        }
    }

    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result_list = new LinkedList<>();
        if (nums.length < 3)
            return result_list;
        // 排序
        Arrays.sort(nums);
        System.out.println(Arrays.toString(nums));
        // 标定正负的分界线
        // 对应的数值要么是0，要么是负数
        int zero_point = -1;
        for (int i = 0; i < nums.length - 1; i++) {
            if (nums[i] <= 0 && nums[i + 1] > 0) {
                zero_point = i;
                break;
            }
        }
        // 全正数或者全负数直接结束
        // it means all numbers are positive or negative
        if (zero_point == -1) {
            return result_list;
        }
        // start algorithm
        // 当搜索到指定位置时没有合适的内容则直接返回，即前元素<0，后元素>0
        for (int i = 0; i <= zero_point; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            for (int j = zero_point + 1; j < nums.length; j++) {
                if (j > 0 && nums[j] == nums[j - 1]) {
                    continue;
                }
                int third_number_index = findThirdNumber(nums, i, j, zero_point);
                if (third_number_index > 0) {
                    List<Integer> sub_result = new LinkedList<>();
                    sub_result.add(nums[i]);
                    sub_result.add(nums[j]);
                    sub_result.add(nums[third_number_index]);
                    result_list.add(sub_result);
                }
            }
        }
        System.out.println(result_list);
        return result_list;
    }

    public int findThirdNumber(int[] nums, int i, int j, int zero_point) {
        int gap = nums[i] + nums[j];
        int index = -1;
        if (gap == 0) {
            return nums[zero_point] == 0 ? zero_point : -1;
        } else if (gap < 0) {
            // 需要在正数集合里寻找，但是不要回头找
            System.out.println(Arrays.toString(Arrays.copyOfRange(nums, j, nums.length)));
            index = Arrays.binarySearch(Arrays.copyOfRange(nums, j, nums.length), -gap);
            index += j;
        } else {
            // 需要在负数的集合里找，不能回头找
            System.out.println(Arrays.toString(Arrays.copyOfRange(nums, i, zero_point)));
            index = Arrays.binarySearch(Arrays.copyOfRange(nums, i, zero_point), -gap);
            index += i;
        }
        return index;
    }

    public String addBinary(String a, String b) {
        // 保证a的长度大于等于b
        if (a.length() < b.length()) {
            String c = a;
            a = b;
            b = c;
        }

        // 从末尾开始，以b为运算长度进行计算，将每次运算的结果插入到result的前部
        StringBuilder result = new StringBuilder();
        int carry = 0;
        int a_index = a.length() - 1;
        int b_index = b.length() - 1;
        // 当前位结果 = a.x + b.x + carry
        for (int i = 0; i < b.length(); i++) {
            int temp = Integer.parseInt(a.substring(a_index, a_index + 1)) +
                    Integer.parseInt(b.substring(b_index, b_index + 1)) + carry;
            a_index--;
            b_index--;
            carry = temp / 2;
            result.insert(0, temp % 2);
        }

        // 继续处理carry和a
        while (carry != 0 && a_index >= 0) {
            int temp = Integer.parseInt(a.substring(a_index, a_index + 1)) + carry;
            carry = temp / 2;
            result.insert(0, temp % 2);
            a_index--;
        }

        // 顶格进位
        if (carry == 1) {
            return "1" + result;
        }

        // carry为0，将a的剩余部分添加到result中
        result.insert(0, a.substring(0, a_index + 1));

        return result.toString();
    }

    public int mySqrt(int x) {
        if (x == 0) {
            return 0;
        }

        int left = 0, right = x, ans = -1;
        while (left <= right) {
            int middle = left + (right - left) / 2;
            if ((long) middle * middle <= x) {
                ans = middle;
                left = middle + 1;
            } else {
                right = middle - 1;
            }
        }
        return ans;
    }

    public ListNode deleteDuplicates(ListNode head) {
        ListNode result = null, current_node = null;
        int last_val = -999;
        while (head != null) {
            if (last_val != head.val) {
                // 非重复元素，添加
                ListNode temp = new ListNode(head.val);
                if (result == null) {
                    result = temp;
                    current_node = result;
                } else {
                    current_node.next = temp;
                    current_node = current_node.next;
                }
                head = head.next;
            }
        }
        return result;
    }

    public void merge(int[] nums1, int m, int[] nums2, int n) {
//        // 保证m和n不为0
//        if (m == 0) {
//            for (int i = 0; i < nums2.length; i++) {
//                nums1[i] = nums2[i];
//            }
//            return;
//        } else if (n == 0) {
//            return;
//        }
//
//        int[] nums3 = new int[m + n];
//        int index1 = 0, index2 = 0, flag = -1;
//        for (int i = 0; i < m + n; ) {
//            nums3[i++] = nums1[index1] > nums2[index2] ? nums2[index2++] : nums1[index1++];
//            if (index1 == m) {
//                // nums1的值已经全部被取完
//                while (index2 != n) {
//                    nums3[i++] = nums2[index2++];
//                }
//                break;
//            } else if (index2 == n) {
//                while (index1 != m) {
//                    nums3[i++] = nums1[index1++];
//                }
//                break;
//            }
//        }
//
//        // 将nums3的值传会给nums1
//        for (int i = 0; i < m + n; i++) {
//            nums1[i] = nums3[i];
//        }
        System.arraycopy(nums2, 0, nums1, m, n);
        Arrays.sort(nums1);
    }

    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> result = new LinkedList<>();

        if (root == null) return result;
        getResult(root, result);

        return result;
    }

    public void getResult(TreeNode root, List<Integer> result) {
        if (root.left != null) {
            getResult(root.left, result);
        }
        result.add(root.val);
        if (root.right != null) {
            getResult(root.right, result);
        }
    }

    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        } else if (p == null || q == null) {
            return false;
        } else if (p.val == q.val) {
            return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
        } else return false;
    }

    // 对称树
    public boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        else return compareLR(root.left, root.right);
    }

    public boolean compareLR(TreeNode root1, TreeNode root2) {
        if (root1 == null && root2 == null) {
            return true;
        } else if (root1 == null || root2 == null) {
            return false;
        } else if (root1.val != root2.val) {
            return false;
        } else {
            return compareLR(root1.left, root2.right) && compareLR(root1.right, root2.left);
        }
    }

    // 二叉树最大深度
    public int maxDepth(TreeNode root) {
        int max = 0;
        if (root == null) {
            return 0;
        } else {
            return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
        }
    }

    // 升序数组转二叉搜索树
    public TreeNode sortedArrayToBST(int[] nums) {
        if (nums.length == 0) {
            return null;
        }
        int middle = nums.length / 2;
        TreeNode root = new TreeNode(nums[middle]);
        root.left = sortedArrayToBST(Arrays.copyOfRange(nums, 0, Math.max(0, middle)));
        root.right = sortedArrayToBST(Arrays.copyOfRange(nums, middle + 1, Math.max(middle + 1, nums.length)));
        return root;
    }

    // 岛屿的数量
    public int numIslands(char[][] grid) {
        int islandCount = 0;

        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                if (grid[i][j] == '1') {
                    islandCount += 1;
                    IslandsBFS(grid, i, j);
                }
            }
        }
        return islandCount;
    }

    // 岛屿数量的BFS
    public void IslandsBFS(char[][] grid, int x, int y) {
        // 初始化并将根元素入队
        Queue<int[]> queue = new LinkedList<>();
        int m = grid.length;
        int n = grid[0].length;
        // 编码
        int[] code = {x, y};
        queue.add(code);
        while (!queue.isEmpty()) {
            code = queue.poll();
            // 解码
            x = code[0];
            y = code[1];
            if (grid[x][y] == '1') {
                // 当前点置0，在重复搜索时会被直接返回，所以无需考虑多次重复访问一个点的问题
                grid[x][y] = '0';
                // 将上下左右放入队列中
                if (x > 0) {
                    code = new int[]{x - 1, y};
                    queue.add(code);
                }
                if (x + 1 < m) {
                    code = new int[]{x + 1, y};
                    queue.add(code);
                }
                if (y > 0) {
                    code = new int[]{x, y - 1};
                    queue.add(code);
                }
                if (y + 1 < n) {
                    code = new int[]{x, y + 1};
                    queue.add(code);
                }
            }
        }
    }

    // 平衡二叉树高度计算
    public int height(TreeNode root) {
        if (root == null) return 0;
        int left_height, right_height;
        if ((left_height = height(root.left)) == -1 ||
                (right_height = height(root.right)) == -1 ||
                Math.abs(left_height - right_height) > 1) {
            return -1;
        } else {
            return Math.max(left_height, right_height) + 1;
        }
    }

    // 最小深度
    public int minDepth(TreeNode root) {
        if (root == null) return 0;
        Queue<TreeNode> queue = new LinkedList<>();
        int height = 1;
        TreeNode temp = new TreeNode(9999);
        temp.left = new TreeNode(9999);
        temp.right = new TreeNode(9999);

        // 广度优先搜索
        queue.add(root);
        queue.add(temp);
        while (!queue.isEmpty()) {
            TreeNode treeNode = queue.poll();
            if (treeNode.left == null && treeNode.right == null) {
                return height;
            } else if (treeNode.val == 9999) {
                queue.add(temp);
                height += 1;
            } else {
                if (treeNode.left != null) queue.add(treeNode.left);
                if (treeNode.right != null) queue.add(treeNode.right);
            }
        }

        return height;
    }

    // 生成杨辉三角
    public List<List<Integer>> generate(int numRows) {
        // 预处理
        List<List<Integer>> lists = new LinkedList<>();
        List<Integer> list = new LinkedList<>();
        list.add(1);
        lists.add(list);
        if (numRows == 1) return lists;
        list.add(1);
        lists.add(list);
        if (numRows == 2) return lists;

        // 第三层开始添加
        for (int i = 2; i < numRows; i++) {
            list.clear();
            List<Integer> temp = lists.get(i - 1);
            list.add(1);
            for (int j = 0; j < temp.size() - 1; j++) {
                list.add(temp.get(j) + temp.get(j + 1));
            }
            list.add(1);
            lists.add(list);
        }

        return lists;
    }

    // 每日温度
    public int[] dailyTemperatures(int[] temperatures) {
        Deque<Integer> main = new LinkedList<>();   // 单调递减栈
        Deque<Integer> assist = new LinkedList<>(); // 用来存储main中各个栈元素的下标位置
        int[] result = new int[temperatures.length];
        main.offerFirst(temperatures[0]);
        assist.offerFirst(0);
        for (int i = 1; i < temperatures.length; i++) {
            while (!main.isEmpty()) {
                // 非空，需要进行比较
                // 当遇到较大数或者空时，元素入栈，continue
                if (temperatures[i] > main.element()) {
                    // 出栈，并计算结果
                    main.pollFirst();
                    int temp_index = assist.pollFirst();
                    result[temp_index] = i - temp_index;
                    // 空栈时不需要继续，直接结束
                    if (main.isEmpty()) {
                        main.offerFirst(temperatures[i]);
                        assist.offerFirst(i);
                        break;
                    }
                } else {
                    main.offerFirst(temperatures[i]);
                    assist.offerFirst(i);
                    break;
                }
            }
        }
        while (!assist.isEmpty()) {
            // 剩余的元素填0
            result[assist.pollFirst()] = 0;
        }
        return result;
    }

    // 逆波兰表达式求值
    public int evalRPN(String[] tokens) {
        Deque<Integer> deque = new LinkedList<>();
        for (String token : tokens) {
            if (token.equals("+")) {
                int num1 = deque.pollFirst();
                int num2 = deque.pollFirst();
                deque.offerFirst(num2 - num1);
            } else if (token.equals("-")) {
                int num1 = deque.pollFirst();
                int num2 = deque.pollFirst();
                deque.offerFirst(num2 - num1);
            } else if (token.equals("*")) {
                int num1 = deque.pollFirst();
                int num2 = deque.pollFirst();
                deque.offerFirst(num2 - num1);
            } else if (token.equals("/")) {
                int num1 = deque.pollFirst();
                int num2 = deque.pollFirst();
                deque.offerFirst(num2 - num1);
            } else {
                deque.offerFirst(Integer.parseInt(token));
            }
        }
        return deque.element();
    }

    // 岛屿的数量-DFS
    public int numIslandsDFS(char[][] grid) {
        int count = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                if (grid[i][j] == '1') {
                    // 将该点置'0'表示已经遍历过
                    // 开始DFS
                    findLandsDFS(grid, i, j);
//                    System.out.println(Arrays.deepToString(grid));
                    count++;
                }
            }
        }
        return count;
    }

    public void findLandsDFS(char[][] grid, int i, int j) {
        if (grid[i][j] == '1') {
            grid[i][j] = '0';
            if (i < grid.length - 1) {
                findLandsDFS(grid, i + 1, j);
            }
            if (i > 0) {
                findLandsDFS(grid, i - 1, j);
            }
            if (j < grid[i].length - 1) {
                findLandsDFS(grid, i, j + 1);
            }
            if (j > 0) {
                findLandsDFS(grid, i, j - 1);
            }
        }
    }

    // 树节点
    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    // 最小栈
    class MinStack {
        Deque<Integer> deque;
        Deque<Integer> dequeAss;

        /**
         * initialize your data structure here.
         */
        public MinStack() {
            deque = new LinkedList<>();
            dequeAss = new LinkedList<>();
        }

        public void push(int val) {
            deque.offerFirst(val);
            if (dequeAss.isEmpty()) {
                dequeAss.offerFirst(val);
            } else {
                dequeAss.offerFirst(Math.min(val, dequeAss.element()));
            }
        }

        public void pop() {
            deque.remove();
            dequeAss.remove();
        }

        public int top() {
            return deque.element();
        }

        public int getMin() {
            return dequeAss.element();
        }
    }

    // 用队列实现栈
    class MyQueue {
        Stack<Integer> stack, backup;

        /** Initialize your data structure here. */
        public MyQueue() {
            stack = new Stack<>();
            backup = new Stack<>();
        }

        /** Push element x to the back of queue. */
        public void push(int x) {
            while (!stack.isEmpty()){
                backup.push(stack.pop());
            }
            backup.push(x);
            while(!backup.isEmpty()){
                stack.push(backup.pop());
            }
        }

        /** Removes the element from in front of queue and returns that element. */
        public int pop() {
            return stack.pop();
        }

        /** Get the front element. */
        public int peek() {
            return stack.peek();
        }

        /** Returns whether the queue is empty. */
        public boolean empty() {
            return stack.isEmpty();
        }
    }

    // 图节点
    class Node {
        public int val;
        public List<Node> neighbors;

        public Node() {
            val = 0;
            neighbors = new ArrayList<Node>();
        }

        public Node(int _val) {
            val = _val;
            neighbors = new ArrayList<Node>();
        }

        public Node(int _val, ArrayList<Node> _neighbors) {
            val = _val;
            neighbors = _neighbors;
        }
    }
}
