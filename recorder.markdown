># May 2023

## 对于目标追踪的理解
MOT multiple object tracking 目标追踪 联系匹配

* 通用评价标准 FP FN IDS MOTA IDF1 HOTA
* 主流评测数据集 MOTchallenge humaninevents BDD100K

## 大致方向
* 基于Tracking-by-detection的DeepSORT
* 基于检测和跟踪联合的MOT
   JDE、FairMOT
* 基于注意力机制的MOT 
   TrackFormer
## 特有名词
* 轨迹(Trajectory)。MOT系统的输出量，一条轨迹对应这一个目标在一个时间段内中的位置序列

* 小段轨迹(Tracklet)。 形成Trajectory过程中的轨迹片段。完整的Trajectory是由属于同一物理目标的Tracklets构成的

# DeepSORT
## 理论
>YoloV5生成bbox

>匈牙利算法 基于Reid iou构建代价矩阵完成tracks和bboxes的匹配

>基于卡尔曼滤波生成预测
1.it is such a general and powerful tool for combining information in the presence of uncertainty.<br>
2.The Kalman filter assumes that both variables (postion and velocity, in our case) are random and Gaussian distributed. Each variable has a mean value \(\mu\), which is the center of the random distribution (and its most likely state), and a variance \(\sigma^2\), which is the uncertainty:
 不同的信息源都存在误差 结合传感器观测值和模型预测值得到最优结果<br>

>代价矩阵(reid,iou,卡尔曼估计)

缺点：缺乏时空信息和运动信息
## 代码
> ![a](Capture.PNG)
# Online Multiple Object Tracking with Cross-Task Synerg  
## 理论
>通过注意和分散解决occlusion

>协同预测步骤和特征提取步骤
一般目标跟踪由两个部分：对每帧进行目标检测和内嵌联系性(a target attention module, a distractor attention module, and an identity-aware memory aggregation.)这篇文章通过注意力机制等联系两边，相互促进形成了协作
## 代码
 >![a](Capture6.PNG)

 >![a](Capture5.PNG)
# A Simple Baseline for Multi-Object Tracking
>重点强调anchor-free的好处，并通过不同层次特征融合


# ByteTrack: Multi-Object Tracking by Associating Every Detection Box








1、如何处理中途出现的新目标

2、如何处理中途消失的目标

3、正确目标关联



# 目标追踪双端整合






# 第一篇：目标追踪
## Tracking-by-Detecton 根据目标检测结果跟踪


# BERT GPT Transformer
# AE

# 注意力机制
>key-query-value
> one-hot vs word embedding
> paf part affinity feild 怎么对向量打标签

# 第二篇 目标追踪的启发



















# 总结
* 我的研究领域:MOT 
* 我将重点解决:occlusion
* 接受论文偏好：提出想法innovative(有启发性)
* 我的方法需要具有:simple but effective