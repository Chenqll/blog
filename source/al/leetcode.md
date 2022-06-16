# leetcode
## 复杂度
- 数组在内存中是连续存储，复杂度为常数；然而链表复杂度为 n。
- 常数操作：与数据量无关的操作
- 时间复杂度：常数操作数量的表达式去除低阶项，和高阶系数。（选择排序的时间复杂度为O(n^2)），指标项同时，直接跑确定优劣
- 空间复杂度：申请的空间

## 排序
- 选择
  
- 冒泡
- 插入
- 归并
- 快速
- 堆
  
## 二叉树
- 运行 c++ 文件的步骤
    ```bash
    g++ name.cpp -o name
    # will generate a name.exe then run it
    name.exe
    ```
### 完全二叉树与满二叉树
- 完全二叉树需从 1 开始标号，并且 Parent=Child/2 
### 二叉树的遍历
- 假设一个 A，B，C 的满二叉树
- 序列遍历
  |从左到右的遍历方式有三种|
  |-----|----|-----|
  |a,b,c|前序|根在前|
  |b,a,c|中序|根在中|
  |b,c,a|后序|根在后|
- 层次遍历与前序一致

### 二叉树题目
#### 二叉树的递归方法：前序 中序 后序
- 前序->刚进入一个节点时执行，从根结点出发 ABC
- 中序->一个结点的左子树遍历完，即将遍历右子树时执行，根在中 BAC
- 后序->将要离开一个结点时执行，根在后 BCA
#### 递归解决问题的方法
**以 DepthLength 为例**：
- 遍历方法
  ```c++
  void traverse(TreeNode* root){
      // 二叉树递归时的必要操作，判断当前结点是否为空
      if (root == NULL){// 指针判断是否为空只能用 `NULL` 关键字 
            return 0;
      }
      depth++;//前序操作
      res=max(depth,res);//这个操作放在哪儿都行
      traverse(root->left);
      traverse(root->right);
      depth--;//
      return depth;
  }
  ```
- 分解方法
  ```c++
  void traverse(TreeNode* root){
      if (root == NULL){
          return 0;
      }
      leftMax=traverse(root->left);
      rightMax=traverse(rott->right);
      return max(leftMax,rightMax)+1;
  }
  ```

#### 综上,遇到一道二叉树的题目时的通用思考过程
- 是否可以通过**遍历**一遍二叉树得到答案？如果可以，用一个 traverse 函数配合外部变量来实现。
- 是否可以定义一个**递归函数**，通过*子问题*（子树）的答案推导出原问题的答案？如果可以，写出这个递归函数的定义，并充分利用这个函数的返回值。
- 无论使用哪一种思维模式，你都要明白二叉树的每一个节点需要做什么，需要在什么时候（前中后序）做。
|二叉树相关题目|
|------|-------|-------|
|前序遍历|中序遍历|后序遍历|
|[最大深度问题](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)|||