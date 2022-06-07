# Framwork of DeepLearning
## op & kernel
- `Kernel` 其实就是对 Tensor 进行操作的一个函数，一段计算逻辑.
- 对于一些简单的 `Kernel`（如"Add"）来说，其是一个纯函数。但对于一些复杂的 `Kernel` 来说，其可能需要初始化一些状态，也有可能拥有一些中间状态需要保存，所以其就不是纯函数。因此，在 TensorFlow 中，用 Class 的形式来表现 Kernel，方便存储中间状态。
- `Op` 就是 `Kernel` 的集合，一个 Op 代表的是有一定共性的多个 Kernel
- 举例而言，在 TensorFlow 中，“MatMul” op 对应的 kernel 是 class MatMulOp<Device, T, USE_CUBLAS>。这个模板类的每一个全特化对应的都是一个真正的 kernel，所以在 "MatMul" op 这个概念之下，其实有着很多的 kernel。例如 CPU 实现的针对于 float 数据类型的 kernel（三个模板参数是 kCPU, float, false ），GPU上使用 cublas实 现的针对 float 的 kernel（kGPU, float, true），GPU 上不用 cublas 实现的针对 half 数据类型的 kernel（kGPU, half, false）等等。
- op 是多个有共性的 kernel 的抽象
## define user op
  ### 在 oneflow/ir/include/OneFlow/OneFlowUserOps.td 定义 op
  ```
  def OneFlow_LeakyReluOp : OneFlow_BaseOp<"leaky_relu", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$alpha
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
    }
  ```
  ### 新增 Op 定义之后，需要重新 make.
  此时会自动在 build 目录下的 oneflow/core/framework/ 目录下生成文件 op_generated.h和 op_generated.cpp ，op_generated.h 负责生成定义 op 时定义的接口，但是接口的实现需要在 oneflow/user/ops/leaky_relu_op.cpp 实现

  ### 实现定义、生成的接口 
  oneflow/user/ops/leaky_relu_op.cpp
  ```
    #include "oneflow/core/framework/framework.h"
    #include "oneflow/core/framework/op_generated.h"

    namespace oneflow {

    /* static */ Maybe<void> LeakyReluOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
    const Shape& x_shape = ctx->InputShape("x", 0);
    Shape* y_shape = ctx->OutputShape("y", 0);
    *y_shape = x_shape;
    return Maybe<void>::Ok();
    }

    /*static*/ Maybe<void> LeakyReluOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
    return InferLogicalTensorDesc(ctx);
    }

    /* static */ Maybe<void> LeakyReluOp::GetSbp(user_op::SbpContext* ctx) {
    const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
    FOR_RANGE(int64_t, i, 0, x_tensor.shape().NumAxes()) {
        ctx->NewBuilder().Split(user_op::OpArg("x", 0), i).Split(user_op::OpArg("y", 0), i).Build();
    }
    return Maybe<void>::Ok();
    }

    /* static */ Maybe<void> LeakyReluOp::InferDataType(user_op::InferContext* ctx) {
    *ctx->OutputDType("y", 0) = ctx->InputDType("x", 0);
    return Maybe<void>::Ok();
    }

    }  // namespace oneflow
    ......
  ```

  ### 实现在 kernel 里的计算逻辑 CPU/GPU
    
  ```
    template<typename T>
    class CpuLeakyReluKernel final : public user_op::OpKernel {
    public:
    CpuLeakyReluKernel() = default;
    ~CpuLeakyReluKernel() = default;

    private:
    void Compute(user_op::KernelComputeContext* ctx) const override {
        const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
        user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
        const int32_t elem_cnt = x->shape().elem_cnt();
        const float alpha = ctx->Attr<float>("alpha");
        const T* x_ptr = x->dptr<T>();
        T* y_ptr = y->mut_dptr<T>();
        FOR_RANGE(int32_t, i, 0, elem_cnt) { y_ptr[i] = x_ptr[i] > 0 ? x_ptr[i] : x_ptr[i] * alpha; }
    }
    bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
    };

  ```

  ### 定义+实现计算逻辑后，对 op 进行注册，使用宏 `REGISTER_USER_KERNEL`
   
    ```
    #define REGISTER_CPU_LEAKY_RELU_KERNEL(dtype)             \
    REGISTER_USER_KERNEL("leaky_relu")                      \
      .SetCreateFn<CpuLeakyReluKernel<dtype>>()           \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu") \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));
    ```

    1. 将op注册到Functional接口，Functional接口通过Python C扩展，使用户能使用Python代码调用该op.
        `oneflow/oneflow/core/functional/impl/activation_functor.cpp`
        ```
        # 注册到Functional接口
        class LeakyReluFunctor {
        public:
        LeakyReluFunctor() {
            op_ = CHECK_JUST(one::OpBuilder("leaky_relu").Input("x").Output("y").Build());
        }
        Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const float& alpha) const {
            MutableAttrMap attrs;
            JUST(attrs.SetAttr<float>("alpha", alpha));
            return OpInterpUtil::Dispatch<one::Tensor>(*op_, {x}, attrs);
        }

        private:
        std::shared_ptr<OpExpr> op_;
        };
        ```
  ### functional 通过 yaml 配置文件，自动帮我们生成接口，在 `oneflow/oneflow/core/functional/functional_api.yaml` 配置接口
    ``` 
    - name: "leaky_relu" # 其中name表示导出到python接口后函数的名字,flow._C.leaky_relu(...)
    signature: "Tensor (Tensor x, Float alpha) => LeakyRelu" # signature表示的是函数的返回值类型，输入类型
    bind_python: True # 表示这个接口是否需要导出到Python接口,比如的leaky_relu_grad，我们不会在python层用到（但会在c++层 求导使用，所以我们设置为False），在C++中可以以 functional::LeakyReluGrad 这种方式进行调用
    ```
