package com.yaowen.myJava;

import org.junit.Assert;

import java.math.BigInteger;
import java.util.*;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Solution {
    /**
     * LC-1220 Count Vowels Permutation
     */
    int vowel_factor = (int) 1e9 + 7;
    private int count = 0;

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

    // 目标和

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

    // 平衡二叉树
    public boolean isBalanced(TreeNode root) {
        // 递归做法，思路：左树平衡+右树平衡+左右树高度差为1
        // 采用自下而上的递归方法可以保证height的调用次数为O(n)次，而自上而下最差的情况为O(n^2)
        // 在计算过程中，用不可及的值进行判定，当某侧高度为不可及值时，则说明该侧不平衡，直接通过短路原则返回所有的递归调用（编程关键）
        return height(root) != -1;
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

    public String findDifferentBinaryString(String[] nums) {
        if (nums == null || nums.length == 0) return null;
        if (nums.length == 1) return String.valueOf((Integer.parseInt(nums[0]) + 1) % 2);
        Set<Integer> set = new HashSet<>();
        for (String s : nums) {
            set.add(Integer.parseInt(s, 2));
        }

        int max = (int) Math.pow(2, nums[0].length()) - 1;
        for (int i = 0; i <= max; i++) {
            if (!set.contains(i)) {
                StringBuilder result = new StringBuilder();
                result.append(Integer.toBinaryString(i));
                while (result.length() <= nums[0].length()) {
                    result.insert(0, "0");
                }
                return result.toString();
            }
        }
        return null;
    }

    public int minimizeTheDifference(int[][] mat, int target) {
        Deque<Integer> queue = new LinkedList<>();
        Deque<Integer> result = new LinkedList<>();
        queue.offer(0);
        result.offer(-target);
        int row = -1;
        while (!queue.isEmpty()) {
            int size = queue.size();
            int bestLineAns = 999999;
            row++;
            while (size-- > 0) {
                queue.remove();
                // 将下一列入栈
                if (row < mat.length) {
                    int last_result = result.poll();
                    for (int num : mat[row]) {
                        int current_result = last_result + num;
                        if (current_result > 0 && Math.abs(current_result) >= bestLineAns) {
                            // 如果当前结果大于0且当前的结果已经不是行最优解时，可以直接抛弃
                            // 因为加法会使得其结果越来越偏离最优解（因为此时是全联接状态）
                            continue;
                        }
                        queue.offer(num);
                        result.offer(current_result);
                        if (Math.abs(current_result) < bestLineAns) {
                            bestLineAns = Math.abs(current_result);
                        }
                    }
                }
            }
        }
        // 最后在所有结果中找出最优解
        int ans = Math.abs(result.poll());
        while (!result.isEmpty()) {
            if (ans > Math.abs(result.element())) {
                ans = Math.abs(result.poll());
            } else {
                result.remove();
            }
        }

        return ans;
    }

    // 中序遍历的非递归写法
    public List<Integer> inorderTraversal2(TreeNode root) {
        if (root == null)
            return new LinkedList<Integer>();
        List<Integer> result = new LinkedList<>();
        Deque<TreeNode> stack = new LinkedList<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode current = stack.peek();
            while ((current = current.left) != null) {
                stack.push(current);
            }
            // 当无右树时不断弹栈
            while (!stack.isEmpty()) {
                current = stack.pop();
                result.add(current.val);
                if (current.right != null) {
                    stack.push(current.right);
                    break;
                }
            }
        }
        return result;
    }

    // 787. K 站中转内最便宜的航班 —— 动态规划
    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
        // index = 0使用，k代表中转次数，边=中转次数+1，所以初始化时需要使用k+2
        // int[k+2][n] => 经过k次中转到达城市n的最小花费
        // int[n][k+2] => 到达城市n经过k次中转所需的最小花费，在解题过程中因为以中转次数作为状态转移的方向，所以该表示方法不如上面一种
        final int PRICE_INFINITY = 10000 * 101 + 1;
        int[][] dp = new int[k + 2][n];
        for (int[] temp : dp) {
            Arrays.fill(temp, PRICE_INFINITY);
        }
        dp[0][src] = 0;
        for (int i = 1; i < k + 2; i++) {
            for (int[] flight : flights) {
                int from = flight[0], to = flight[1], cost = flight[2];
                dp[i][to] = Math.min(dp[i][to], dp[i - 1][from] + cost);
            }
        }
        int ans = PRICE_INFINITY;
        for (int i = 1; i < k + 2; i++) {
            if (dp[i][dst] < ans) {
                ans = dp[i][dst];
            }
        }
        return ans == PRICE_INFINITY ? -1 : ans;
    }

    // LC-03
    public int lengthOfLongestSubstring(String s) {
        int temp = 0, max = 0;
        if (s == null || s.length() == 0)
            return 0;
        if (s.length() == 1)
            return 1;

        // <char, times>
        Map<Character, Integer> map = new HashMap<>();
        char[] chars = s.toCharArray();

        int head = 0, tail = 0;
        while (tail < s.length()) {
            // 当前节点有效
            if (map.get(chars[tail]) == null || map.get(chars[tail]) == 0) {
                map.put(chars[tail], 1);
                temp++;
                if (temp > max) {
                    max = temp;
                }
            } else {
                // 当前节点无效，需要不断循环搜索，直到当前字符有效
                map.put(chars[tail], 2);
                temp++;
                while (map.get(chars[tail]) == 2) {
                    map.put(chars[head], map.get(chars[head]) - 1);
                    head++;
                    temp--;
                }
            }
            tail++;
        }
        return max;
    }

    public int findKthLargest(int[] nums, int k) {
        quickSort(nums);
        return nums[nums.length - k];
    }

    public void quickSort(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        partition(nums, left, right);
    }

    // 787. K 站中转内最便宜的航班 - 广度优先搜索 - 超限
//    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
//        List<List<int[]>> ordered_flights = getOrderedFlights(flights);
//        Deque<Integer> costs = new LinkedList<>();
//        Deque<int[]> queue = new LinkedList<>();
//        List<Integer> result = new ArrayList<>();
//        // 记忆化搜索
//        int[][] memo = new int[flights.length][flights[0].length];
//        for (int[] flight : flights) {
//            if (flight[0] == src) {
//                // [src, des, price]
//                queue.add(flight);
//                costs.add(0);
//            }
//        }
//        while (!queue.isEmpty() && k-- >= 0) {
//            int size = queue.size();
//            while (size-- > 0) {
//                int[] current_line = queue.poll();
//                int current_cost = costs.poll();
//                current_cost = current_cost + current_line[2];
//                if (current_line[1] == dst) {
//                    // 到达目的地
//                    result.add(current_cost);
//                } else if (k >= 0) {
//                    // 将后续的路线压入栈中
//                    for (int[] flight : getNextLine(current_line[1], ordered_flights)) {
//                        queue.add(flight);
//                        costs.add(current_cost);
//                    }
//                }
//            }
//        }
//
//        if (!result.isEmpty()) {
//            Integer[] results = result.toArray(new Integer[0]);
//            Arrays.sort(results);
//            return results[0];
//        } else {
//            return -1;
//        }
//    }
//
//    public List<List<int[]>> getOrderedFlights(int[][] flights) {
//        // 下标下面挂着相同的src的链表
//        List<List<int[]>> arrayList = new ArrayList<>();
//        // 需要多初始化一个
//        for (int i = 0; i < 101; i++) {
//            List<int[]> list = new ArrayList<>();
//            arrayList.add(i, list);
//        }
//        for (int[] flight : flights) {
//            int src = flight[0];
//            arrayList.get(src).add(flight);
//        }
//        return arrayList;
//    }
//
//    public int[][] getNextLine(int des, List<List<int[]>> ordered_flights) {
//        int size = ordered_flights.get(des).size();
//        int[][] next_lines = ordered_flights.get(des).toArray(new int[size][3]);
//        return next_lines;
//    }

    public void partition(int[] nums, int left, int right) {
        if (nums.length == 1 || left == right)
            return;
        int key = nums[left];
        int p_left = left, p_right = right;
        // 相等时相当于需要把key放到对应的位置上，然后继续分治
        while (p_left < p_right) {
            while (p_left < p_right && nums[p_right] > key) {
                p_right--;
            }
            nums[p_left++] = nums[p_right];
            while (p_left < p_right && nums[p_left] <= key) {
                p_left++;
            }
            nums[p_right--] = nums[p_left];
        }
        nums[p_left] = key;
        partition(nums, left, p_left - 1);
        partition(nums, p_right + 1, right);
    }

    public int findKthLargest_MaxHeap(int[] nums, int k) {
        // 建堆
        buildHeap(nums);
        // 删除K - 1次头节点
        for (int i = 0; i < k; i++) {
            deleteHeapTop(nums);
        }
        return nums[0];
    }

    public void buildHeap(int nums[]) {
        for (int i = nums.length / 2; i >= 0; i--) {
            maxHeapify(nums, i);
        }
    }

    public int[] deleteHeapTop(int nums[]) {
        nums[0] = nums[nums.length - 1];
        nums = Arrays.copyOfRange(nums, 0, nums.length - 1);
        for (int i = nums.length / 2; i >= 0; i--) {
            maxHeapify(nums, i);
        }
        return nums;
    }

    public void maxHeapify(int nums[], int index) {
        int left = 2 * index + 1, right = 2 * index + 2;
        if (left < nums.length - 1 && nums[left] > nums[index]) {
            int temp = nums[index];
            nums[index] = nums[left];
            nums[left] = temp;
        }
        if (right < nums.length - 1 && nums[right] > nums[index]) {
            int temp = nums[index];
            nums[index] = nums[right];
            nums[right] = temp;
        }
    }

    // LC-56 合并区间
    // 基本解法：1.排序，2.按条件更新或者追加区间
    public int[][] merge(int[][] intervals) {
        if (intervals.length == 0) {
            return null;
        }
        Arrays.sort(intervals, (o1, o2) -> o1[0] - o2[0]);
        List<int[]> result = new LinkedList<>();
        result.add(intervals[0]);
        for (int i = 1; i < intervals.length; i++) {
            int l = intervals[i][0], r = intervals[i][1];
            if (l > result.get(result.size() - 1)[1]) {
                // 新元素
                result.add(new int[]{l, r});
            } else {
                result.get(result.size() - 1)[1] = Math.max(result.get(result.size() - 1)[1], r);
            }
        }
        return result.toArray(new int[result.size()][]);
    }

    // LC-53 最大子序和
    public int maxSubArray(int[] nums) {
        return getArrayInfo(nums, 0, nums.length - 1).mSum;
    }

    public Status getArrayInfo(int[] nums, int left, int right) {
        if (left == right) {
            return new Status(nums[left], nums[left], nums[left], nums[left]);
        }
        int middle = (left + right) >> 1;
        Status leftStatus = getArrayInfo(nums, left, middle - 1);
        Status rightStatus = getArrayInfo(nums, middle, right);
        return combineResult(leftStatus, rightStatus);
    }

    public Status combineResult(Status left, Status right) {
        int tSum = left.tSum + right.tSum;
        int lSum = Math.max(left.lSum, left.tSum + right.lSum);
        int rSum = Math.max(right.rSum, right.tSum + left.rSum);
        int mSum = Math.max(left.rSum + right.lSum, Math.max(left.mSum, right.mSum));
        return new Status(lSum, rSum, mSum, tSum);
    }

    public boolean containsDuplicate(int[] nums) {
        Map<String, Object> map = new HashMap<>();
        Set<Integer> set = new HashSet<>();
        for (int i : nums) {
            if (!set.add(i))
                return true;
        }
        return false;
    }

    /**
     * LC-498 对角线遍历
     */
    public int[] findDiagonalOrder(int[][] mat) {
        int x = 0, y = 0;
        int length = mat.length - 1;
        boolean flag = true;
        int[] result = new int[mat.length * mat.length];
        int index = 0;
        while (x <= length || y <= length) {
            System.out.println(Arrays.toString(result));
            if (flag) {
                result[index++] = mat[y--][x++];
                if (y < 0 || x > length) {
                    y = 0;
                    flag = false;
                    while (x > length) {
                        x--;
                        y++;
                    }
                }
            } else {
                System.out.println("x=" + x + "  y=" + y);
                result[index++] = mat[y++][x--];
                if (x < 0 || y > length) {
                    x = 0;
                    flag = true;
                    while (y > length) {
                        y--;
                        x++;
                    }
                }
            }
        }

        return result;
    }

    /**
     * LC-566 重塑矩阵
     */
    public int[][] matrixReshape(int[][] mat, int r, int c) {
        if (mat == null) {
            return null;
        } else if (mat.length * mat[0].length != r * c) {
            return mat;
        }

        int[][] result = new int[r][c];
        int mc = mat[0].length;

        for (int index = 0; index < r * c; index++) {
            result[index / c][index % c] = mat[index / mc][index % mc];
        }

        return result;

    }

    public String reverseWords(String s) {
        StringBuilder sb = new StringBuilder();
        String[] ss = s.split(" ");
        arrayReverser(ss);
        for (String temp : ss) {
            temp = temp.replace(" ", "");
            if (temp.equals("")) {
                continue;
            }
            sb.append(temp);
            sb.append(" ");
        }
        return sb.deleteCharAt(sb.length() - 1).toString();
    }

    public <T> void arrayReverser(T[] a) {
        int left = 0, right = a.length - 1;
        while (left < right) {
            T temp = a[left];
            a[left++] = a[right];
            a[right--] = temp;
        }
    }

    public int totalMoney(int n) {
        int wholeWeeks = n / 7;
        int restDays = n % 7;
        Double result = 3.5 * wholeWeeks * wholeWeeks + 24.5 * wholeWeeks + restDays * (wholeWeeks + 1.0) + (restDays - 1.0) * restDays / 2.0;
        return result.intValue();
    }

    public int countVowelPermutation(int n) {
        int factor = (int) 1e9 + 7;
        int a = 1, e = 1, i = 1, o = 1, u = 1;
        while (--n != 0) {
            int aa = e + i + u;
            int ee = a + i;
            int ii = e + o;
            int oo = i;
            int uu = i + o;
            a = aa;
            e = ee;
            i = ii;
            o = oo;
            u = uu;
        }
        return (a + e + i + o + u) % factor;
    }

//    public int getVowelPermutationCount(int n, String curr) {
//        int count = 0;
//        int lastIndex = curr.length() - 1;
//        if (n != 0) {
//            if (curr.charAt(lastIndex) == 'a') {
//                count += getVowelPermutationCount(n - 1, curr + "e");
//            } else if (curr.charAt(lastIndex) == 'e') {
//                count += getVowelPermutationCount(n - 1, curr + "a");
//                count += getVowelPermutationCount(n - 1, curr + "i");
//            } else if (curr.charAt(lastIndex) == 'i') {
//                count += getVowelPermutationCount(n - 1, curr + "a");
//                count += getVowelPermutationCount(n - 1, curr + "e");
//                count += getVowelPermutationCount(n - 1, curr + "o");
//                count += getVowelPermutationCount(n - 1, curr + "u");
//            } else if (curr.charAt(lastIndex) == 'o') {
//                count += getVowelPermutationCount(n - 1, curr + "i");
//                count += getVowelPermutationCount(n - 1, curr + "u");
//            } else {
//                count += getVowelPermutationCount(n - 1, curr + "a");
//            }
//        } else {
//            count = 1;
//        }
//
//        return count % vowel_factor;
//    }

    public List<Integer> getTimeList(List<String> timePoints) {
        List<Integer> timeList = new ArrayList<>();
        for (String time : timePoints) {
            String[] splitTime = time.split(":");
            timeList.add(Integer.parseInt(splitTime[0]) * 60 + Integer.parseInt(splitTime[1]));
            timeList.add(1440 + Integer.parseInt(splitTime[0]) * 60 + Integer.parseInt(splitTime[1]));
        }
        return timeList;
    }

    public int findMinDifference(List<String> timePoints) {
        if (timePoints.size() > 1440) {
            return 0;
        }

        List<Integer> list = getTimeList(timePoints);
        Collections.sort(list);

        int min = 1440;
        for (int i = 0; i < list.size() - 1; i++) {
            if (list.get(i + 1) - list.get(i) < min) {
                min = list.get(i + 1) - list.get(i);
            }
        }

        return min;
    }

    public int secondMinimum(int n, int[][] edges, int time, int change) {

        List<Integer>[] map = new List[n + 1];
        boolean[] valid = new boolean[n + 1];
        boolean[] visited = new boolean[n + 1];
        for (int i = 0; i <= n; i++) {
            map[i] = new LinkedList<Integer>();
            valid[i] = true;
            visited[i] = false;
        }
        for (int[] edge : edges) {
            map[edge[0]].add(edge[1]);
            map[edge[1]].add(edge[0]);
        }

        // 每个点最多遍历两次
        Set<Integer> set = new HashSet<>();
        set.add(1);
        int count = 0;
        int level = 0;
        boolean flag = false;
        while (count < 2) {
            level++;
            Set<Integer> temp = new HashSet<>();
            for (int index : set) {
                if (valid[index]) {
                    if (!visited[index]) {
                        temp.addAll(map[index]);
                        if (temp.size() >= n)
                            break;
                        if (flag && temp.contains(n))
                            break;
                        visited[index] = true;
                    } else {
                        valid[index] = false;
                    }

                }
            }
            if (temp.contains(n)) {
                count++;
                flag = true;
            } else if (flag) {
                level = level + 1;
            }
            set = temp;
        }

        int result = 0;
        while (level-- > 0) {
            result += time;
            if (level != 0 && (result / change) % 2 != 0) {
                result = change * (result / change + 1);
            }
        }

        return result;
    }

    public int numberOfWeakCharacters(int[][] properties) {
        int count = 0;
        int index = 1;
        Arrays.sort(properties, (a1, a2) -> a2[0] - a1[0]);
        int[] compareTarget = new int[]{properties[0][0], properties[0][1]};
        int[] nextCompareTarget = null;

        // 初始化compareTarget
        while (index < properties.length && compareTarget[0] == properties[index][0]) {
            if (properties[index][1] > compareTarget[1]) {
                compareTarget[1] = properties[index][1];
            }
            index++;
        }

        for (int i = index; i < properties.length; i++) {
            if (properties[i][0] != properties[i - 1][0] && nextCompareTarget != null) {
                compareTarget = nextCompareTarget;
                nextCompareTarget = null;
            }
            // 判断防御力大小，小的直接淘汰，大的比较是否能够成为nextCompareTarget
            if (properties[i][1] < compareTarget[1]) {
                count++;
            } else {
                if (nextCompareTarget == null) {
                    nextCompareTarget = properties[i];
                } else {
                    if (nextCompareTarget[1] < properties[i][1]) {
                        nextCompareTarget = properties[i];
                    }
                }
            }
        }
        return count;
    }

    public String convert(String s, int numRows) {
        if (numRows == 1) {
            return s;
        }

        List<List<Character>> lists = new ArrayList<>();
        int index = 0;
        boolean flag = false;
        for (int i = 0; i < numRows; i++) {
            lists.add(new LinkedList<>());
        }

        for (Character c : s.toCharArray()) {
            lists.get(index).add(c);
            if (index == 0) {
                index++;
                flag = false;
            } else if (index == numRows - 1) {
                index--;
                flag = true;
            } else {
                if (flag) {
                    index--;
                } else {
                    index++;
                }
            }
        }

        StringBuilder sb = new StringBuilder();
        for (List<Character> list : lists) {
            for (char c : list.toArray(new Character[0])) {
                sb.append(c);
            }
        }

        return sb.toString();
    }

    public String intToRoman(int num) {
        int[] values = new int[]{1, 4, 5, 9, 10, 40, 50, 90, 100, 400, 500, 900, 1000};
        String[] symbols = new String[]{"I", "IV", "V", "IX", "X", "XL", "L", "XC", "C", "CD", "D", "CM", "M"};
        StringBuilder sb = new StringBuilder();
        while (num != 0) {
            for (int i = values.length - 1; i >= 0; i--) {
                if (num >= values[i]) {
                    sb.append(symbols[i]);
                    num = num - values[i];
                    break;
                }
            }
        }
        return sb.toString();
    }

    public int[][] highestPeak(int[][] isWater) {
        if (isWater == null) {
            return null;
        }

        int m = isWater.length, n = isWater[0].length;
        int[][] result = new int[m][n];
        int[][] offsets = new int[][]{{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        Deque<int[]> deque = new LinkedList<>();

        for (int[] r : result) {
            Arrays.fill(r, Integer.MAX_VALUE);
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (isWater[i][j] == 1) {
                    deque.offerLast(new int[]{i, j});
                    result[i][j] = 0;
                }
            }
        }

        while (!deque.isEmpty()) {
            int size = deque.size();
            while (size-- > 0) {
                int[] p = deque.poll();
                System.out.println(Arrays.toString(deque.toArray(new int[0][0])));
                for (int[] offset : offsets) {
                    assert p != null;
                    int x = p[0] + offset[0], y = p[1] + offset[1];
                    if (x > 0 && x < m - 1 && y > 0 && y < n - 1) {
                        if (result[x][y] == Integer.MAX_VALUE) {
                            deque.offerLast(new int[]{x, y});
                            result[x][y] = result[p[0]][p[1]] + 1;
                        }
                    }
                }
            }
        }

        return result;
    }

//    该算法超时
//    public int[][] highestPeak(int[][] isWater) {
//        if (isWater == null) {
//            return null;
//        }
//
//        int m = isWater.length, n = isWater[0].length;
//        int[][] result = new int[m][n];
//        for (int[] r : result) {
//            Arrays.fill(r, Integer.MAX_VALUE);
//        }
//        List<int[]> waterMap = new ArrayList<>();
//        for (int i = 0; i < m; i++) {
//            for (int j = 0; j < n; j++) {
//                if (isWater[i][j] == 1) {
//                    waterMap.add(new int[]{i, j});
//                    result[i][j] = 0;
//                }
//            }
//        }
//
//        for (int i = 0; i < m; i++) {
//            for (int j = 0; j < n; j++) {
//                if (isWater[i][j] == 0) {
//                    for (int[] waterPosition : waterMap) {
//                        int distance = Math.abs(i - waterPosition[0]) + Math.abs(j - waterPosition[1]);
//                        if (distance < result[i][j])
//                            result[i][j] = distance;
//                    }
//                }
//            }
//        }
//
//        return result;
//    }

    public int[][] processWater(int i, int j, int[][] result) {
        if (i > 0) {
            result[i - 1][j] = 1;
        }
        if (i < result.length - 1) {
            result[i + 1][j] = 1;
        }
        if (j > 0) {
            result[i][j - 1] = 1;
        }
        if (j < result[i].length - 1) {
            result[i][j + 1] = 1;
        }

        return result;
    }

    public String[] uncommonFromSentences(String s1, String s2) {
        String[] strs1 = s1.split(" "), strs2 = s2.split(" ");
        Map<String, Integer> map1 = new HashMap<>();
        Map<String, Integer> map2 = new HashMap<>();

        for (String s : strs1) {
            map1.put(s, map1.getOrDefault(s, 0) + 1);
        }
        for (String s : strs2) {
            map2.put(s, map2.getOrDefault(s, 0) + 1);
        }

        List<String> result = new LinkedList<>();
        for (String key : map1.keySet()) {
            if (map1.get(key) == 1 && !map2.containsKey(key))
                result.add(key);
        }
        for (String key : map2.keySet()) {
            if (map2.get(key) == 1 && !map1.containsKey(key))
                result.add(key);
        }

        return result.toArray(new String[0]);
    }

    public String longestNiceSubstring(String s) {
        int maxPos = 0, maxLen = 0;
        for (int i = 0; i < s.length(); i++) {
            int lower = 0, upper = 0;
            for (int j = i; j < s.length(); j++) {
                char c = s.charAt(j);
                if (Character.isLowerCase(c)) {
                    lower = lower | 1 << c - 'a';
                } else {
                    upper = upper | 1 << c - 'A';
                }
                if (lower == upper && j - i + 1 > maxLen) {
                    maxPos = i;
                    maxLen = j - i + 1;
                }
            }
        }
        return s.substring(maxPos, maxPos + maxLen);
    }

    public String reversePrefix(String word, char ch) {
        int index = word.indexOf(ch);
        if (index != -1) {
            StringBuilder sb = new StringBuilder();
            sb.append(word, 0, index + 1);
            sb.reverse();
            sb.append(word, index + 1, word.length());
            return sb.toString();
        }
        return word;
    }

    public List<String> letterCombinations(String digits) {
        if (digits.length() == 0)
            return new LinkedList<>();

        String[] ss = new String[]{"abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        int offset = 2;
        List<String> result = new LinkedList<>();
        String init = "";
        result.add(init);

        for (char c : digits.toCharArray()) {
            int index = Integer.parseInt(String.valueOf(c)) - offset;
            int length = ss[index].length();
            List<String> temp = result;
            result = new LinkedList<>();
            for (int i = 0; i < length; i++) {
                for (String s : temp) {
                    result.add(s + String.valueOf(ss[index].charAt(i)));
                }
            }
        }

        return result;
    }

    // LC-8 字符串转化
    // 其它思路：自动机、
    public int myAtoi(String s) {
        if (s == null || s.length() == 0)
            return 0;

        int validIndex = 0;
        int finalResult = 0;
        int MAX = 2147483647;
        int MIN = -2147483648;
        boolean negative = false;
        for (char c : s.toCharArray()) {
            if (c == ' ')
                validIndex++;
            else
                break;
        }

        if (validIndex < s.length() && s.charAt(validIndex) == '-') {
            negative = true;
            validIndex++;
        } else if (validIndex < s.length() && s.charAt(validIndex) == '+') {
            validIndex++;
        }

        // System.out.println(negative + "   " + validIndex);

        while (validIndex < s.length() && Character.isDigit(s.charAt(validIndex))) {
            if (!negative && (finalResult > MAX / 10 || finalResult == MAX / 10 && s.charAt(validIndex) > '7')) {
                return MAX;
            } else if (negative && (finalResult > MAX / 10 || finalResult == MAX / 10 && s.charAt(validIndex) > '8')) {
                return MIN;
            }
            finalResult = finalResult * 10 + s.charAt(validIndex++) - '0';
        }

        if (negative)
            return -finalResult;
        else
            return finalResult;
    }

    public List<List<Integer>> threeSum(int[] nums) {
        if (nums == null || nums.length < 3)
            return new LinkedList<>();

        List<List<Integer>> result = new LinkedList<>();
        Set<ThreeSumClass> set = new HashSet<>();

        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; i++) {
            for (int j = i + 1; j < nums.length - 1; j++) {
                int k = Arrays.binarySearch(nums, j + 1, nums.length, -(nums[i] + nums[j]));
                if (k > j) {
                    set.add(new ThreeSumClass(nums[i], nums[j], nums[k]));
                }
            }
        }

        for (ThreeSumClass a : set) {
            result.add(a.toList());
        }

        return result;
    }

    public int hammingWeight(int n) {
        System.out.println(Integer.bitCount(10));
        return 0;
    }

    public int networkBecomesIdle(int[][] edges, int[] patience) {
        int[] sp = getSP(edges);
        int[] times = computeTotalTime(sp, patience);
        int maxTime = -1;
        for (int time : times) {
            if (time > maxTime)
                maxTime = time;
        }
        return maxTime + 1;
    }

    public int[] getSP(int[][] edges) {
        // 计算每个节点到达节点0的最短路径的长度
        Map<Integer, Queue<Integer>> map = new HashMap<>();
        for (int[] edge : edges) {
            if (map.containsKey(edge[0])) {
                map.get(edge[0]).add(edge[1]);
            } else {
                Queue<Integer> queue = new LinkedList<>();
                queue.add(edge[1]);
                map.put(edge[0], queue);
            }
            if (map.containsKey(edge[1])) {
                map.get(edge[1]).add(edge[0]);
            } else {
                Queue<Integer> queue = new LinkedList<>();
                queue.add(edge[0]);
                map.put(edge[1], queue);
            }
        }

        // 利用BFS来实现最短路径的查找
        int[] sp = new int[map.size()];
        boolean[] visited = new boolean[map.size()];
        visited[0] = true;
        int currLevel = 0;
        Queue<Integer> queue = map.get(0);
        while (!queue.isEmpty()) {
            currLevel++;
            int size = queue.size();
            while (size-- > 0) {
                if (visited[queue.peek()])
                    continue;
                sp[queue.peek()] = currLevel;
                visited[queue.peek()] = true;
                Queue<Integer> tempQueue = map.get(queue.poll());
                while (!tempQueue.isEmpty()) {
                    queue.add(tempQueue.poll());
                }
            }
        }
        return sp;
    }

    public int[] computeTotalTime(int[] sp, int[] patience) {
        int[] totalTime = new int[sp.length];
        for (int i = 1; i < sp.length; i++) {
            int offset = ((2 * sp[i]) % patience[i]) == 0 ? -1 : 0;
            totalTime[i] = (2 * sp[i] / patience[i] + offset) * patience[i] + 2 * sp[i];
        }
        return totalTime;
    }

    public boolean winnerOfGame(String colors) {
        if (colors.length() <= 2)
            return false;

        int aCount = getCount(colors, 'A');
        int bCount = getCount(colors, 'B');
        return aCount - bCount > 0;
    }

    public int getCount(String colors, char target) {
        char[] cs = colors.toCharArray();
        int count = 0, currCount = 0;

        for (int i = 0; i < colors.length(); i++) {
            if (currCount < 2 && cs[i] == target) {
                currCount++;
            } else if (cs[i] == target) {
                count++;
            } else if (cs[i] != target) {
                currCount = 0;
            }
        }

        return count;
    }

    public boolean hasAlternatingBits(int n) {
        int test = 1;
        int last = (n & test) == 1 ? 0 : 1;
        while (n != 0) {
            if ((n & test) == last) {
                return false;
            }
            last = n & test;
            n = n >> 1;
        }
        return true;
    }

    public boolean canReorderDoubled(int[] arr) {
        Arrays.sort(arr);
        Map<Integer, Integer> map = new LinkedHashMap<>();
        for (int a : arr) {
            map.put(a, map.getOrDefault(a, 0) + 1);
        }
        for (int k : map.keySet()) {
            int doubleK = 2 * k;
            if (map.containsKey(doubleK)) {
                int kValue = map.get(k);
                int doubleKValue = map.get(doubleK);
                int newKValue = doubleKValue > kValue ? 0 : kValue - doubleKValue;
                int newDoubleKValue = newKValue == 0 ? doubleKValue - kValue : 0;
                map.put(k, newKValue);
                map.put(doubleK, newDoubleKValue);
                System.out.println(map);
            }
        }
        for (int k : map.keySet()) {
            if (map.get(k) != 0)
                return false;
        }

        return true;
    }

    public int divide(int dividend, int divisor) {
        boolean rev = (dividend >> 31 ^ divisor >> 31) == 0;  // true = positive
        if (dividend == 0) {
            return 0;
        } else if (dividend == Integer.MIN_VALUE) {
            if (divisor == 1) {
                return Integer.MIN_VALUE;
            } else if (divisor == -1) {
                return Integer.MAX_VALUE;
            }
        }

        if (divisor == Integer.MIN_VALUE) {
            return dividend == Integer.MIN_VALUE ? 1 : 0;
        }

        int left = 1, right = dividend;
        int mid = left + ((right - left) >> 1);
        int ans = 0;
        while (left <= right) {
            if (divideChecker(dividend, divisor, mid)) {
                ans = mid;
                if (mid == Integer.MAX_VALUE) {
                    break;
                }
                left = left + 1;
            } else {
                right = right + 1;
            }
        }
        return rev ? ans : -ans;
    }

    public boolean divideChecker(int dividend, int divisor, int n) {
        int result = 0, add = divisor;
        while (n != 0) {
            System.out.println(n);
            if ((n & 1) != 0) {
                if (result < dividend - add)
                    return false;
                result += add;
            }
            if (n != 1) {
                if (add < dividend - add)
                    return false;
                add += add;
            }
            n = n >> 1;
        }
        return true;
    }

    public int lengthLongestPath(String input) {
        if (!input.contains(".")) {
            return 0;
        }

        input = "/" + input;

        input = input.replace("\n", "\n\t")
                .replace("\n", "")
                .replace("\t", "/");
        int nextLevel = 0;
        int currentLevel = 0;
        int max = 0;
        StringBuilder sb = new StringBuilder();
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < input.length(); i++) {
            char curr = input.charAt(i);
            if (curr != '/') {
                sb.append(curr);
            } else {
                list.add(sb.length());
                nextLevel++;
                while (input.charAt(i + 1) == '/') {
                    nextLevel++;
                    i++;
                }

                if (nextLevel <= currentLevel) {
                    if (sb.indexOf(".") != -1) {
                        max = Math.max(getNewLength(list), max);
                    }
                    while (list.size() > nextLevel) {
                        list.remove(list.size() - 1);
                    }
                }

                sb = new StringBuilder();
                currentLevel = nextLevel;
                nextLevel = 0;
            }
        }

        if (sb.indexOf(".") != -1) {
            list.add(sb.length());
            max = Math.max(getNewLength(list), max);
        }

        return max;
    }

    public int getNewLength(List<Integer> list) {
        int result = 0;
        for (int n : list) {
            result += n;
        }
        return result + list.size() - 2;
    }

    public static class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
            next = null;
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

    class ThreeSumClass {
        int i, j, k;

        ThreeSumClass(int i, int j, int k) {
            this.i = i;
            this.j = j;
            this.k = k;
        }

        List<Integer> toList() {
            List<Integer> list = new LinkedList<>();
            list.add(i);
            list.add(j);
            list.add(k);
            return list;
        }

        // Hash是否相等的关键在于定义Equals
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            ThreeSumClass that = (ThreeSumClass) o;
            return i == that.i && j == that.j && k == that.k;
        }

        @Override
        public int hashCode() {
            return Objects.hash(i, j, k);
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

        /**
         * Initialize your data structure here.
         */
        public MyQueue() {
            stack = new Stack<>();
            backup = new Stack<>();
        }

        /**
         * Push element x to the back of queue.
         */
        public void push(int x) {
            while (!stack.isEmpty()) {
                backup.push(stack.pop());
            }
            backup.push(x);
            while (!backup.isEmpty()) {
                stack.push(backup.pop());
            }
        }

        /**
         * Removes the element from in front of queue and returns that element.
         */
        public int pop() {
            return stack.pop();
        }

        /**
         * Get the front element.
         */
        public int peek() {
            return stack.peek();
        }

        /**
         * Returns whether the queue is empty.
         */
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

    class LRUCache {
        LRUNode head, tail;
        Map<Integer, LRUNode> map;
        int maxSize;

        public LRUCache(int capacity) {
            head = new LRUNode(-1, -1);
            tail = new LRUNode(-2, -2);
            head.after = tail;
            tail.before = head;
            map = new HashMap<>();
            maxSize = capacity;
        }

        public int get(int key) {
            LRUNode node = map.get(key);
            if (node == null) {
                return -1;
            } else {
                refresh(node);
                return node.val;
            }
        }

        public void put(int key, int value) {
            // 插入head，淘汰tail
            // 键存在，替换
            if (map.containsKey(key)) {
                LRUNode node = map.get(key);
                node.val = value;
                refresh(node);
            } else {
                // 键不存在，先插入再选择性删除
                LRUNode node = new LRUNode(key, value);
                node.before = head;
                node.after = head.after;
                head.after = node;
                node.after.before = node;
                map.put(key, node);
                if (map.size() > maxSize) {
                    LRUNode temp = tail.before;
                    tail.before = temp.before;
                    temp.before.after = tail;
                    temp.after = null;
                    temp.before = null;
                    map.remove(temp.key);
                }
            }
        }

        public void refresh(LRUNode node) {
            node.before.after = node.after;
            node.after.before = node.before;
            node.before = head;
            node.after = head.after;
            head.after = node;
            node.after.before = node;
        }

        class LRUNode {
            int val;
            int key;    // 在删除的时候用于同时删除hashmap中的值使用
            LRUNode before;
            LRUNode after;

            public LRUNode(int K, int V) {
                key = K;
                val = V;
                before = null;
                after = null;
            }
        }
    }

    /**
     * LC-53 最大子序和的辅助类
     */
    class Status {
        // 左侧开始最大子序和、右侧开始最大子序和、区间最大子序和、区间和
        int lSum, rSum, mSum, tSum;

        public Status(int l, int r, int m, int t) {
            this.lSum = l;
            this.rSum = r;
            this.mSum = m;
            this.tSum = t;
        }
    }

    class NumArray {
        int[] tree;
        int[] nums;

        public NumArray(int[] nums) {
            this.nums = nums;
            tree = new int[nums.length + 1];
            for (int i = 0; i < nums.length; i++) {
                add(i + 1, nums[i]);
            }
        }

        public void update(int index, int val) {
            add(index + 1, val - nums[index]);
            nums[index] = val;
        }

        public int sumRange(int left, int right) {
            return prefixSum(right + 1) - prefixSum(left);
        }

        public int lowBit(int x) {
            return x & -x;
        }

        public void add(int index, int val) {
            while (index < tree.length) {
                tree[index] += val;
                index = index + lowBit(index);
            }
        }

        public int prefixSum(int index) {
            int sum = 0;
            while (index > 0) {
                sum += tree[index];
                index -= lowBit(index);
            }
            return sum;
        }
    }

    class RandomizedSet {
        Map<Integer, Integer> map;
        int[] array;
        Random random;
        int index;

        public RandomizedSet() {
            map = new HashMap<>();
            array = new int[8];
            index = 0;
            random = new Random();
        }

        public boolean insert(int val) {
            if (map.containsKey(val)) {
                return false;
            }
            if (index >= array.length) {
                int[] temp = new int[array.length * 2];
                System.arraycopy(array, 0, temp, 0, index);
                array = temp;
            }
            map.put(val, index);
            array[index++] = val;
            return true;
        }

        public boolean remove(int val) {
            if (map.containsKey(val)) {
                int targetIndex = map.get(val);
                array[targetIndex] = array[index - 1];
                map.put(array[targetIndex], targetIndex);
                map.remove(val);
                index--;
                return true;
            }
            return false;
        }

        public int getRandom() {
            int randomIndex = random.nextInt(index);
            return array[randomIndex];
        }
    }

    class MakeSquare {
        public boolean makesquare(int[] matchsticks) {
            if (matchsticks.length < 4)
                return false;

            Arrays.sort(matchsticks);
            for (int i = 0, j = matchsticks.length - 1; i < j; i++, j--) {
                int temp = matchsticks[i];
                matchsticks[i] = matchsticks[j];
                matchsticks[j] = temp;
            }

            int perimeter = 0, sideLength = 0;
            perimeter = Arrays.stream(matchsticks).sum();
            sideLength = perimeter / 4;
            if (perimeter % 4 != 0 || matchsticks[matchsticks.length - 1] > sideLength)
                return false;

            int[] edges = new int[]{0, 0, 0, 0};
            return validator(matchsticks, 0, sideLength, edges);
        }

        public boolean validator(int[] matchsticks, int index, int sideLength, int[] edges) {
            if (index == matchsticks.length)
                return true;

            for (int i = 0; i < edges.length; i++) {
                edges[i] += matchsticks[index];
                if (edges[i] <= sideLength && validator(matchsticks, index + 1, sideLength, edges)) {
                    return true;
                }
                edges[i] -= matchsticks[index];
            }

            return false;
        }
    }

    public TreeNode deleteNode(TreeNode root, int key) {
        TreeNode curr = root;
        TreeNode father = null;
        while (curr != null) {
            if (curr.val == key) {
                curr = buildNewBST(curr.left, curr.right);
                if (father != null) {
                    if (father.left != null && father.left.val == key) {
                        father.left = curr;
                    } else {
                        father.right = curr;
                    }
                } else {
                    root = curr;
                }
                return root;
            }

            if (key < curr.val) {
                father = curr;
                curr = curr.left;
            } else {
                father = null;
                curr = curr.right;
            }
        }
        return root;
    }

    public TreeNode buildNewBST(TreeNode left, TreeNode right) {
        if (left == null)
            return right;
        if (right == null)
            return left;

        TreeNode temp = left;
        TreeNode tempFather = null;
        while (temp.left != null || left.right != null) {
            tempFather = temp;
            if (temp.left != null) {
                temp = temp.left;
            } else {
                temp = temp.right;
            }
        }

        if (tempFather.left == temp) {
            tempFather.left = null;
        } else {
            tempFather.right = null;
        }

        temp.left = left;
        temp.right = right;
        TreeNode resultRoot = temp;

        while (true) {
            if (temp.left != null && temp.left.val > temp.val) {
                int swap = temp.val;
                temp.val = temp.left.val;
                temp.left.val = swap;
                temp = temp.left;
            } else if (temp.right != null && temp.right.val < temp.val) {
                int swap = temp.val;
                temp.val = temp.right.val;
                temp.right.val = swap;
                temp = temp.right;
            } else {
                break;
            }
        }

        return resultRoot;
    }

    static class MyLinkedList {
        int size;
        Node head;

        public MyLinkedList() {
            size = 0;
            head = new Node();
        }

        public int get(int index) {
            if (index >= size)
                return -1;

            Node curr = head.next;
            while (index-- > 0) {
                curr = curr.next;
            }
            return curr.val;
        }

        public void addAtHead(int val) {
            head.next = new Node(val, head.next);
            size++;
        }

        public void addAtTail(int val) {
            Node curr = head;
            while (curr.next != null)
                curr = curr.next;

            curr.next = new Node(val);
            size++;
        }

        public void addAtIndex(int index, int val) {
            if (index == size) {
                addAtTail(val);
                return;
            }

            if (index > size)
                return;

            Node before = head, curr = head.next;
            while (index-- > 0) {
                before = curr;
                curr = curr.next;
            }

            before.next = new Node(val, curr);
            size++;
        }

        public void deleteAtIndex(int index) {
            if (index > size - 1)
                return;

            Node before = head, curr = head.next;
            while (index-- > 0) {
                before = curr;
                curr = curr.next;
            }

            before.next = curr.next;
            curr.next = null;
            size--;
        }

        static class Node {
            public int val;
            public Node next;

            public Node() {
            }

            ;

            public Node(int val) {
                this.val = val;
                next = null;
            }

            public Node(int val, Node next) {
                this.val = val;
                this.next = next;
            }
        }
    }

    public int numUniqueEmails(String[] emails) {
        Set<String> emailSet = new HashSet<>();
        for (String email : emails) {
            String[] parts = email.split("@");
            parts[0] = parts[0].split("\\+")[0].replace(".", "");
            String newAddress = parts[0] + "@" + parts[1];
            emailSet.add(newAddress);
        }
        return emailSet.size();
    }

    public ListNode detectCycle(ListNode head) {
        Set<ListNode> set = new HashSet<>();
        while (head != null) {
            if (set.contains(head))
                return head;
            set.add(head);
            head = head.next;
        }

        Random random = new Random();
        random.nextDouble();

        return null;
    }

    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null)
            return true;

        ListNode slow = head, fast = head.next;
        ListNode head1, head2 = null;
        while (fast != null) {
            slow = slow.next;
            if (fast.next != null) {
                fast = fast.next.next;
                if (fast == null) {
                    head2 = slow.next;
                }
            } else {
                head2 = slow;
                fast = null;
            }
        }

        ListNode before = null, curr = head, later;
        while (curr != null && curr != slow) {
            later = curr.next;
            curr.next = before;
            before = curr;
            curr = later;
        }

        head1 = before;
        while (head1 != null && head2 != null) {
            if (head1.val != head2.val)
                return false;
            head1 = head1.next;
            head2 = head2.next;
        }

        return true;
    }

    static class MyCalendarThree {
        private final TreeMap<Integer, Integer> cnt;

        public MyCalendarThree() {
            cnt = new TreeMap<>();
        }

        public int book(int start, int end) {
            int ans = 0;
            int maxBook = 0;
            cnt.put(start, cnt.getOrDefault(start, 0) + 1);
            cnt.put(end, cnt.getOrDefault(end, 0) - 1);
            for (Map.Entry<Integer, Integer> entry : cnt.entrySet()) {
                int freq = entry.getValue();
                maxBook += freq;
                ans = Math.max(maxBook, ans);
            }
            return ans;
        }
    }

    public int minEatingSpeed(int[] piles, int h) {
        if (piles == null || piles.length == 0)
            return 0;

        int total = 0;
        for (int pile : piles) {
            total += pile;
        }

        return getMinSpeed(piles, h, 0, total);
    }

    public int getMinSpeed(int[] piles, int h, int minSpeed, int maxSpeed) {
        if (minSpeed > maxSpeed)
            return Integer.MAX_VALUE;

        int midSpeed = (minSpeed + maxSpeed) / 2;

        int eatingHours = 0;
        for (int pile : piles) {
            eatingHours += pile / midSpeed;
            if (pile % midSpeed != 0) {
                eatingHours += 1;
            }
        }

        if (minSpeed == maxSpeed && eatingHours > h)
            return Integer.MAX_VALUE;

        if (eatingHours > h) {
            return getMinSpeed(piles, h, midSpeed + 1, maxSpeed);
        } else {
            return Math.min(midSpeed, getMinSpeed(piles, h, minSpeed, midSpeed - 1));
        }
    }

    class RandomPointInNonOverlappingRectangels {
        private Random random;
        private TreeMap<Integer, int[]> map;
        private int totalArea;

        public RandomPointInNonOverlappingRectangels(int[][] rects) {
            random = new Random();
            map = new TreeMap<>();
            totalArea = Arrays.asList(rects).stream().mapToInt(x -> (x[3] - x[1]) * (x[2] - x[0])).sum();

            int sum = 0;
            for (int[] rect : rects) {
                sum += (rect[3] - rect[1]) * (rect[2] - rect[0]);
                map.put(sum, rect);
            }
        }

        public int[] pick() {
            int areaIndex = random.nextInt(totalArea + 1);
            int[] rect = map.ceilingEntry(areaIndex).getValue();


            int x = rect[0] + random.nextInt(rect[2] - rect[0] + 1);
            int y = rect[1] + random.nextInt(rect[3] - rect[1] + 1);
            return new int[]{x, y};
        }
    }

    class DoubleLinkedNode {
        public int val;
        public DoubleLinkedNode prev;
        public DoubleLinkedNode next;
        public DoubleLinkedNode child;
    }

    public int minFlipsMonoIncr(String s) {
        if (s.length() == 1)
            return 0;

        int[][] opt = new int[s.length() + 1][2];
        opt[0][0] = 0;
        opt[0][1] = 0;

        for (int i = 1; i <= s.length(); i++) {
            char c = s.charAt(i - 1);
            if (c == '0') {
                opt[i][0] = opt[i - 1][0];
                opt[i][1] = Math.min(opt[i - 1][0], opt[i - 1][1]) + 1;
            } else {
                opt[i][0] = opt[i - 1][0] + 1;
                opt[i][1] = Math.min(opt[i - 1][0], opt[i - 1][1]);
            }
        }

        return Math.min(opt[s.length()][0], opt[s.length()][1]);
    }
}

