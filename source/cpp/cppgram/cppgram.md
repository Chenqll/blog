# cpp
## general
- 运行 c++ 文件的步骤
    ```bash
    g++ name.cpp -o name
    # will generate a name.exe then run it
    name.exe
    ```
### malloc 函数
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