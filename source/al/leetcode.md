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
### 1、完全二叉树与满二叉树
- 完全二叉树需从 1 开始标号，并且 Parent=Child/2 
### 2、二叉树的遍历
- 假设一个 A，B，C 的满二叉树
- 序列遍历
  |从左到右的遍历方式有三种|
  |-----|----|-----|
  |a,b,c|前序|根在前|
  |b,a,c|中序|根在中|
  |b,c,a|后序|根在后|
- 层次遍历与前序一致

### 3、二叉树题目
#### 3.1 二叉树的递归方法：前序 中序 后序
- 前序->刚进入一个节点时执行，从根结点出发 ABC
- 中序->一个结点的左子树遍历完，即将遍历右子树时执行，根在中 BAC
- 后序->将要离开一个结点时执行，根在后 BCA
#### 3.2 递归解决问题的方法
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

#### 3.3 综上,遇到一道二叉树的题目时的通用思考过程
- 是否可以通过**遍历**一遍二叉树得到答案？如果可以，用一个 traverse 函数配合外部变量来实现。
- 是否可以定义一个**递归函数**，通过*子问题*（子树）的答案推导出原问题的答案？如果可以，写出这个递归函数的定义，并充分利用这个函数的返回值。
- 无论使用哪一种思维模式，你都要明白二叉树的每一个节点需要做什么，需要在什么时候（前中后序）做。
- **前序遍历**
  ```c++
  List<Integer> preorderTraverse(TreeNode* root){
    List<Integer> res = new List<Integer>();
    if (root == NULL){
      return 0;
    }
    res.add(root->val);
    res.addAll(preorderTraverse(root->left));
    res.addAll(preorderTraverse(roo->right));
    return res;
  }
  ```
- **层序遍历**
  ```c++
  
  ```

#### 3.4 刷题：
- 1. 二叉树的深度问题 leetcode 104
  ```c++
  viod traverse(TreeNode* root){
    if (root == NULL){
      return 0;
    }
    int leftMax=traverse(root->left);
    int rightMax=traverse(root->right);
    return max(leftMax,rightMax)+1;//一个后序遍历，在离开一个结点时进行操作
  }
  ```
- 2. 二叉树的直径问题 leetcode 543
  - 每个结点最大左子树 depth + 最大右子树 depth。
  - 最大子树 depth 与二叉树的 depth 深度一致，可以直接用。
  ```c++
  viod dia(TreeNode* root){
    if (root == NULL){
      return 0;
    }
    int leftMax=traverse(root->left);
    int rightMax=traverse(root->right);
    return leftMax+rightMax;
  }
  viod traverse(TreeNode* root){
    int leftMax=traverse(root->left);
    int rightMax=traverse(root->right);
    return max(leftMax,rightMax)+1;

  }
  ```
## 动态规划
### General
base case -> 状态 -> 选择

**以凑零钱问题** 讲解 General 解法：
```c++
int coinChange(int[] coins,int n){
  return dp(coins,n);
}
// 状态为 int[] coins,int amount;
// 选择为 int coin:coins 状态由于选择的改变而改变
int dp(int[] coins,int amount){
  if(amount == 0) return 0;
  if(amount==-1) return -1;
  for (int coin:coins){
    int subproblem=dp(coins,amount-coin);
    if (subproblem == -1)continue;
    res=min(res,subproblem+1);
  }
  return res==Integer.MAX?-1:res;

}
```

**以斐波那契数列**为例 讲解 memo 常用方法：
```c++
int fin(int n){
  if (n==0 || n==1)return n;//base case
  return fin(n-1)+fin(n-2)//fn=fn-1+fn-2 状态转移方程 
}

// 加 memo 数组减少时间复杂度：
int fin(int n){
  int memo[]=new int[n+1];
  return helper(memo,n);
}
int helper(int[] memo,int n){
  if (n==0||n==1)return n;
  if(memo[n]!=0) return memo[n];
  memo[n]=helper(memo,n-1)+helper(memo,n-2);
  return memo[n];
}
```
## 回溯问题
## 链表算法
### 技巧1-虚拟头结点
以**链接两个有序链表**为例：
```c++
// 定义虚拟头结点 作为最后返回的结果链表
// 定义 P 指针，作为活动指针
// 其实就是 dummy 头结点的地址指向了 p，p 向后扩充的时候，dummy 作为头结点，只要 dummy->next 就能得到后面 p 链接的单链表了 
ListNode mergeTwoLists(ListNode* list1,ListNode* list2){
  // 定义一个虚拟头结点 dummy
  ListNode* dummy=ListNode(-1);
  // 将 dummy 的首地址赋值给 p，p 不断向后向后扩充该链表
  ListNode* p=dummy;
  ListNode* p1=list1;
  ListNode* p2=list2;
  // 链表有两个操作
  // 1. 将 p1 结点赋值给 p->next=p1 ，此时 p1 结点的地址域只是被赋值。
  // 2. 结点滑动 `p=p->next`，此时将下一节点的地址值赋值给 p，此时才能算是将 p1 结点加入到 dummy 中了。
  while(p1 != NULL && p2 != NULL){
    if(p1->val > p2->val){
      p->next=p2;
      p2->next=p2;
    }else{
      p->next=p1;
      p1->next=p1;
    }
    p=p->next;
  }
  if(p1 == NULL){
    p->next=p2;
  }
  if (p2 ==NULL){
    p->next=p1;
  }
  return dummy->next;
  
}
```
**分割链表**：
```c++
ListNode* partition(ListNode* head,int x){
  //分别开辟两个虚拟头结点，分别存储小于x，和大于等于x的 p
  // 返回多少个链表就需要多少个虚拟头结点 dummy
  ListNode* dummy1=new ListNode(-1);
  ListNode* dummy2=new ListNode(-1);
  // 两个分别的p动点，多少个 dummy 就需要多少个子动点
  ListNode* p1;
  ListNode* p2;

  p1=dummy1；
  p2=dummy2;

  //总动点遍历整个链表，一个链表需要一个动点进行连接，
  ListNode* p=head;

  // 使用 while 循环，保证p结点往下遍历
  while(p!=NULL){
    if(p->val>=x){
      p2->next=p;
      p2=p2->next;
    }else{
      p1->next=p;
      p1=p1->next;
    }
    // 解脱 p 结点，p要在源链表往后走
    //??????????????????????????????
    ListNode* temp=p->next;
    p->next=NULL;
    p=temp;
  }
  //要获取一个结点后的链表结构，需要用 dummy->next 获得
  p1->next=dummy2->next;
  return dummy2->next;
}
```
### 递归反转链表
```c++
ListNode* reverse(ListNode* head){
  // base case 用一个 `||` 来进行基础判断
  // 压栈后的基础单元
  if(head == NULL || head->next == NULL){
    return head;
  }
  // 不要压栈，按照函数的定义进行解释，reverse(head->next) 的意思就是反转第一个节点后的链表
  ListNode* last = reverse(head->next);
  // 链表指针赋值 next 指针指向一个地址，应将一个地址 head 赋值给它
  head->next->next=head;
  head->next=NULL;
  return last;
}
```
### 回文链表
- 可以用 **后序遍历** 的方式压栈，得到链表的 last 指针，fore 和 last 两端相遇的方法判断
  ```c++
  ListNode* fore;
  boo isPali(ListNode* head){
     fore=head;
    return traverse(head);
  }
  bool traverse(ListNode* head){
    ListNode* last=head;
    // 递归 三大步，第一步：base case
    if (last == NULL)return true;
    bool res=traverse(last->next);
    // 后序遍历ing
    // 必须和前面的result进行与运算
    res=res&&fore->val==last->val;
    fore=fore->next;
    return res;
  }
  ```
## 数组算法
### 技巧 1-双指针
- 只要数组有序，应该想到双指针技巧
- 以 **删除有序数组的重复项** 为例：
  ```c++
  //分析题目：
  // 1.有序数组，即重复的元素会临近
  
  int remove(vector<int>& nums){
    // 快慢指针的方法可以减少空间复杂度
    if(nums.size()<2)return nums.size();
    int j=0;
    // i 一直在增加，它是快指针，则 j 是慢指针
    for (int i=0;i<nums.size();i++){
      if(nums[i]!=nums[j]){
        nums[i]=nums[++j];
      }
    }
    return i++;
  }
  ```
- **移除元素**
  ```c++
  // 移除数组中的元素，要求返回 int 类型，该int 数据为 vector 向量数组前 int 数个的意思
  int removeElement(vector<int>& nums,int val){
    // 
    int slow=0,fast=0;
    while(fast<nums.sie()){
      if(nums[fast]!=val){
        nums[slow]=nums[fasy];
        slow++;
      }
      fast++;

    }
    return slow;
  }
  ```
- **移动元素**
  ```c++
  //在 移除元素 的基础上，移除该元素后(比如移除0元素)，再将对应的元素加入特定位置
  //需要返回一整个num，此时的返回值为 void
  void moveVal(vector<int>* nums){
    int p = removeVal(nums,0);
    for(;p<nums.size();p++){
      nums[p]=0;
    }
  int removeVal(vector<int>* nums,int val){
    ... 同上
  }
  }
  ```
- **反转字符串**
  ```c++
  void reverseString(vector<char>* s){
    //左右互换，则判断条件应为 while(left<right)

  }
  ```
- **最长的回文字符**：
  ```c++
  
  ```
- **链表内的快慢指针**
  ```c++
  ListNode* deleteDuplicateElemet(ListNode* head){
    // 赋值快慢指针
    ListNode* slow=head;
    ListNode* fast=head;
    // 链表的 basecase
    if(head==NULL)return NULL;
    // 链表快慢指针的退出循环的方式
    while(fast!=NULL){
      if(slow->val!=fast->val){
        // 链表赋值的方式
        slow->next=fast;
        slow=slow->next;
      }
      fast=fast->next;
    }
    // 记得给链表最后断开
    slow->next=NULL;
    return head;
  }
  ```
### 技巧 2 - 前缀和数组
### 技巧 3 - 差分数组
### 技巧 4 - 翻转数组
#### 沿对角线翻转
- 翻转二维矩阵的重点是 “如何将行变为列，如何将列变为行”，能轻松做到这一点的只有沿对角线翻转
  ```c++
  void rotate(vecotr<vector<int>>& matrix){
    // 使用 vector 表示二维数组的方式为 “vector<vector<int>>& matrix” -- 注意 & 符号不能遗漏
    for (int i =0;i<matrix.size();i++){
      // 此时 j 一定要从 i 开始，从 0 开始的话 会将原先调换的数字又调换回来
      for(int j=i;j<matrix.size();j++){
        swap(matrix[i][j],matrix[j][i]);
      }
    }
  }
  ```
#### 行内翻转
- 实现 reverse 函数，实现行内翻转
  ```c++
  // 在二维数组内 如何调用 reverse 函数接口
  ...
  for(vector<int>& x :matrix){
    reverse(x);
  }
  ...

  void reverse(vector<int>& arr){
    // 双指针实现
  }
  ```
