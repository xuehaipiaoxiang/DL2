> April 2023
MOT multiple object tracking 目标追踪 检测+联系匹配
* 评价标准 FP FN IDS MOTA
1. 基于Tracking-by-detection的MOT<br>
   DeepSORT
2. 基于检测和跟踪联合的MOT
   JDE、FairMOT
3. 基于注意力机制的MOT 
   TrackFormer
# A Simple Baseline for Multi-Object Tracking
# ByteTrack: Multi-Object Tracking by Associating Every Detection Box
# Online Multiple Object Tracking with Cross-Task Synerg  
>通过注意和分散解决occlusion
协同预测步骤和特征提取步骤

# Deep SORT
>YoloV5生成bbox
>匈牙利算法 基于Reid iou构建代价矩阵完成tracks和bboxes的匹配
>基于卡尔曼滤波生成预测
>代价矩阵(reid,iou,卡尔曼估计)
缺点：缺乏时空信息和运动信息




1、如何处理中途出现的新目标

2、如何处理中途消失的目标

3、正确目标关联



# 目标追踪双端整合


1.it is such a general and powerful tool for combining information in the presence of uncertainty.<br>
2.The Kalman filter assumes that both variables (postion and velocity, in our case) are random and Gaussian distributed. Each variable has a mean value \(\mu\), which is the center of the random distribution (and its most likely state), and a variance \(\sigma^2\), which is the uncertainty:
 不同的信息源都存在误差 结合传感器观测值和模型预测值得到最优结果<br>

卡尔曼滤波融合<br>
#
>匈牙利算法KM

# 第一篇：目标追踪
## Tracking-by-Detecton 根据目标检测结果跟踪

## 一般目标跟踪由两个部分：对每帧进行目标检测和内嵌联系性(a target attention module, a distractor attention module, and an identity-aware memory aggregation.)这篇文章通过注意力机制等联系两边，相互促进形成了协作
# BERT GPT Transformer
# AE

# 注意力机制
>key-query-value
> one-hot vs word embedding
> paf part affinity feild 怎么对向量打标签

# 第二篇 目标追踪的启发



 ![a](Capture.PNG)




