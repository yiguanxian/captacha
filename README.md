# captacha
get_codepng.py includes all helper functions which can help you complete the task of identification captacha efficiently.                                            

步骤：

1.图片处理 - 对图片进行降噪、二值化处理

2.切割图片 - 将图片切割成单个字符并保存，使用时间戳命名

3.人工标注 - 对切割的字符图片进行人工标注(借助label_data()函数标注会很高效)，作为训练集

4.训练数据 - 用KNN算法训练数据

5.检测结果 - 用上一步的训练结果识别新的验证码 

6.在测试集上测试得到测试集的准确率

7.反思与总结：
（1）使用类似于训练集的验证码识别率挺高，准确率在92%左右
（2）使用其他类型的验证码识别率低，原因是：训练集数据缺乏多样性，量不够大

参考：
https://zhuanlan.zhihu.com/p/43092916
