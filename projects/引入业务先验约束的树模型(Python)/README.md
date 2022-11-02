# GBDT_Simple_Tutorial（梯度提升树简易教程）
## 简介
利用python实现GBDT算法的回归、二分类以及多分类，将算法流程详情进行展示解读并可视化，便于读者庖丁解牛地理解GBDT。

## 项目进度：
- [x] 回归 
- [x] 二分类 
- [x] 多分类
- [x] 可视化 
***
**算法原理以及公式推导请前往blog：**[GBDT算法原理以及实例理解](https://blog.csdn.net/zpalyq110/article/details/79527653)
***
## 依赖环境
- 操作系统：Windows/Linux
- 编程语言：Python3
- Python库：pandas、PIL、pydotplus，
 其中pydotplus库会自动调用Graphviz，所以需要去[Graphviz官网](https://graphviz.gitlab.io/_pages/Download/Download_windows.html)下载`graphviz的-2.38.msi`
，先安装，再将安装目录下的`bin`添加到系统环境变量，此时如果再报错可以重启计算机。详细过程不再描述，网上很多解答。

## 文件结构
- | - GBDT 主模块文件夹
- | --- gbdt.py 梯度提升算法主框架
- | --- decision_tree.py 单颗树生成，包括节点划分和叶子结点生成
- | --- loss_function.py 损失函数
- | --- tree_plot.py 树的可视化
- | - example.py 回归/二分类/多分类测试文件


## 运行指南
- 回归测试：

    `python example.py --model = regression`
- 二分类测试：

    `python example.py --model = binary_cf`
- 多分类测试：

    `python example.py --model = multi_cf`
- 其他可配置参数：`lr` -- 学习率,   `trees` -- 构建的决策树数量即迭代次数,    
`depth` -- 决策树的深度,   `count` -- 决策树节点分裂的最小数据数量,
`is_log` -- 是否打印树的生成过程, `is_plot` -- 是否可视化树的结构.
- 结果文件： 运行后会生成`results`文件夹,里面包含了每棵树的内部结构和生成日志


## 结果展示
仅展示最后所有树的集合，具体每棵树的详细信息望读者自行运行代码~
<img src="https://github.com/Freemanzxp/GBDT_Simple_Tutorial/raw/master/展示图片/all_trees.png"/>