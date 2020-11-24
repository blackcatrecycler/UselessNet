# 使用说明

*Power by Recycler Blacat*

*recently update 2020-11-24*

## 介绍

只有用魔法才能打败魔法，推翻万恶的GPU利用率警察

## 环境

* python<3.8
* torch>1.1.0
* screen
* ubuntu

## 使用说明

### 无脑使用

请开启一个screen窗口，然后`sh go.sh`

### 配置

#### 执行时间

本程序涉及2个时间
* 定时执行时间:请修改go.sh中的 -n后参数（秒）
* 程序运行时间:请修改start.sh中 --time参数(秒)

#### gpu内存占用
全部都在start.sh中修改

|参数|含义|
|:---|:---|
|bnum|block数量，注意这里数量级是n^2|
|inpf|模型中运行的feature层数|
|batch|随机生成的input的batch数|




