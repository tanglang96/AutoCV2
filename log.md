dataset1

take valid_data的时候shuffle

valid_num增加到512

学习率减为1/10

搞定dataset1



学习率有点太小了，速度慢了一些

学习率改为1/2.5（对2345等简单的数据集影响比较大，速度掉下去了），valid_num改为1024（对精度影响比较大），阈值改为0.8

![1564106195530](C:\Users\16600\AppData\Roaming\Typora\typora-user-images\1564106195530.png)

第三个数据集表现较差，主要还是train效果很好，但是val上面不太行，看log像是overfit了，因为数据集太小

**在最后一层增加dropout，防止overfit**

​	主要是针对第三个数据集只有1000多数据量的情况，0.5的dropout还不太够，可能需要加大一点或者更换augmentation的策略，发现加大了效果会更差，需要改变数据增强策略

**将学习率的factor增加到0.75，train发散的时候改为0.25**

这个效果还不错，可以先保留了

**再将resnet18剪枝掉一半layer**

也就第三个会好一点，其他的速度也没有快多少

**还是用原来的resnet18，bs加大到128**

128和64效果并不好，还是改回32吧

**对于第一个数据集，没有augmentation的时候大概1.5s/epoch，有数据增强大概3s/epoch，尝试将dataloader换成dali的模式**

改写成dali比较困难，考虑去掉augmentation看看性能

**将测试集整个加载到cache**

并没有加速，反而会变慢

**设置shape的上限，主要看data1的表现**

并没有下降，说明这样做没有问题

**去掉augmentation**

对第一个数据集影响不大，但是对4，5影响还是有的，结论就是还需要保留数据增强，但是需要一种更高效的处理方式

**将fast augmentation的搜索次数减小为一半**

效果一般，影响不大，还是改为100

**非扫描式的dataloader**

如果不扫描的话，那么每次都会比较慢，效果反而不好，需要找一找更加高效的扫描或者转换方式

**比PIL更快的augmentation，目前看是可以将pipeline写成dali的模式**	

需要注意的地方：

1，内存/显存爆炸

2，flip对于结果的影响





