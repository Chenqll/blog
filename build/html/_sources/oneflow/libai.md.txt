# libai project合并标准
## 摸索和打通出“社区人员往 libai 提交 project 的流程规范”
- load dataset
```python
# fmt:off
VOCAB_URL = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/bert-base-chinese-vocab.txt" # noqa
QQP_TRAIN_URL = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/QQP/train.tsv" # noqa
QQP_TEST_URL = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/QQP/dev.tsv" # noqa
```
- load model 
  
- code规范
  
## 流程
- 主要修改config,model,dataset文件，其中，config.py里直接引用model，dataset放入配置中。

  ```python

  config.py

  from projects.token_classification.model.model import ModelForSequenceClassification
  from projects.token_classification.dataset import CnerDataset
  ...
  ...

  dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(CnerDataset)(
            task_name="cner",
            data_dir="/workspace/CQL_BERT/libai/projects/token_classification/data/cner/cner",
            tokenizer=tokenization.tokenizer,
            max_seq_length=512,
            mode="train",
        ),
    ],
    num_workers=4,
   )
   ...
   ...

  model = LazyCall(ModelForSequenceClassification)(cfg=model_cfg)
  ...
  ...
  ```
- dataset.py文件
· 直接被dataloader引用，需要返回feature对象，通过DataLoader的__getitem__()方法迭代dataset的feature对象

· 通过引入 cner_convert_examples_to_features,cner_processors，将数据转换为可迭代的feature对象
```python
...
self.features = clue_convert_examples_to_features(
                examples,
                tokenizer,
                max_length=max_seq_length,
                pattern=pattern,
                label_list=label_list,
                output_mode=self.output_mode,
            )
...
  def __getitem__(self, i):
        feature = self.features[i]
        return Instance(
            input_ids = DistTensorData(flow.tensor(feature.input_ids, dtype=flow.long)),
            attention_mask = DistTensorData(flow.tensor(feature.attention_mask, dtype=flow.long)),
            token_type_ids = DistTensorData(flow.tensor(feature.token_type_ids, dtype=flow.long)),
            labels = DistTensorData(flow.tensor(feature.labels, dtype=flow.long)),
        )
...
...
```
- dataset_utils.py dataset的主要修改文件，通过不同的任务，设计不同的数据processor，生成example。
```python
def get_labels(self):
        """See base class."""
        return ["X",'B-CONT','B-EDU','B-LOC','B-NAME','B-ORG','B-PRO','B-RACE','B-TITLE',
                'I-CONT','I-EDU','I-LOC','I-NAME','I-ORG','I-PRO','I-RACE','I-TITLE',
                'O','S-NAME','S-ORG','S-RACE',"[START]", "[END]"]
...
def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-','I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            examples.append(InputExample(guid=guid, text_a=text_a,text_b=None, label=labels))
        return examples
```