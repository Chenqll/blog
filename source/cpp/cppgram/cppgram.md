# cpp
## general
- 运行 c++ 文件的步骤
    ```bash
    g++ name.cpp -o name
    # will generate a name.exe then run it
    name.exe
    ```
### malloc 函数
- C++ 中的malloc() 函数将一块未初始化的内存分配给一个指针。它在cstdlib 头文件中定义。
  ```c++
    // allocate memory of int size to an int pointer
    int* ptr = (int*) malloc(sizeof(int));
  ```
## 结构体
- https://blog.csdn.net/qq_51534890/article/details/118637720
### 1.general
```c++
struct student
{
	//成员列表
	string name;		//姓名
	int age;			//年龄
	int score;			//分数
}stu3;                  // 别名
...
```

```c++
struct student stu1;//关键字 struct 可以不写
stu1.name //结构体变量使用操作符"."访问成员
```
### 2.结构体数组
```c++
	//结构体数组
	struct student arr[3] =
	{
		{ "张三", 18, 80 },
		{ "李四", 19, 60 },
		{ "王五", 18, 70 }
	};
    ...
    ...
    arr[i].name//提取数组元素
```
### 3.结构体指针
```c++
struct student stu = { "张三", 18, 100 };

struct student *p= &stu;//使用指针指向结构体 stu 的地址

stu->age=19//指针通过->操作符可以访问成员,通过地址赋值改变 stu 内的数据
```
### 4.结构体做函数参数
- 将结构体作为参数向函数中传递，分为 **值传递**和**地址传递**。
- 值传递和地址传递的示例
  ```c++
    //函数声明
    void printfStudent(student stu);
    void printfStudent2(student* stu);
    ....
    //值传递
    void printfStudent(student stu)
    {
        stu.age = 28;
    }

    //地址传递
    void printfStudent2(student* stu)
    {
        stu->age = 28;
    }
  ```
## Vector
- vector https://blog.csdn.net/m0_61548909/article/details/124215519
### Intro
- 可变大小数组的序列容器。
- 使用 vector
  ```c++
    std::vector<int> first; // empty vector of ints
		std::vector<int> second(4, 100); // four ints with value 100
		std::vector<int> third(second.begin(), second.end()); // iterating through second
		std::vector<int> fourth(third); // a copy of third
  ```
## 单链表 LinkedList
node->next 指向下一 node 的地址，当返回值为 node->next 时，返回的是以node 为头结点的链表结构
## 二叉堆
- https://labuladong.gitee.io/algo/2/21/62/
### 应用场景
- 堆排序
- 优先级队列
### General
- 二叉堆在逻辑上其实是一种特殊的二叉树（完全二叉树），只不过存储在数组里，其中 `parent = arr[i];leftChild = arr[i*2];rightChild = arr[i*2 + 1]`
- 二叉堆分为最大堆和最小堆，其中最大堆的性质是：每个节点都大于它的两个子结点
- 注意数组的第 0 个元素空着不用，所以 size=size+1 并且 arr[1] 作为整颗树的根的话，max=arr[1]
- 主要实现两个 api `sink下沉` & `swim上浮` 
  **API-swim**的实现：
  ```c++
  // 上浮第 x 个元素
  private void swim(int x){
    // 使用 while 保证不断下沉,到达根后停止
    while(x>1 && less(parent(x),x)){
      swap(parent(x),x);
      // 保证插入的数不断下降
      x=parent(x);
    }
  }
  ```
  **API-sink** 的实现：
  ```c++
  private void sink(int x){
    while(left(x)<=size){
      // 先假设左边节点较大
      int max = left(x);
      // 如果右边节点存在，比一下大小
      if (right(x) <= size && less(max, right(x)))
          max = right(x);
      // 结点 x 比俩孩子都大，就不必下沉了
      if (less(max, x)) break;
      // 否则，不符合最大堆的结构，下沉 x 结点
      swap(x, max);
      x = max;
    }
  }
  ```
### 基于二叉堆实现的数据结构-**优先级队列**
- 功能：插入或者删除元素的时候，元素会自动排序，这底层的原理就是二叉堆的操作。
- 主要 API `insert` 和 `delMax`
  **API-insert** 的实现：
  ```c++
  public void insert(int x){
    // size在二叉堆里不仅表示size 还表示最后一个元素
    size++;
    arr[size]=x;
    swim(size);
  }
  ```
  **API-delMax**的实现，主要是先将 arr[1] 放到最后一位后删除，再将刚放入arr[1]的元素进行下沉
  ```c++
  public void delMax(){
    //需要返回删除的最大值
    int max=arr[1];
    swap(1,size);
    arr[size]=null;
    size--;
    sink(1);
    return max;
  }
  ```
