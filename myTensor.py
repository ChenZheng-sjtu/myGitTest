#！/home/zheng/anaconda3/envs/tensorflow/bin/python


# 导入依赖库
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)  #read_data_sets,检查目录下有没有想要的数据，没有就下载，然后解压

# 创建回归模型 
x = tf.placeholder(tf.float32,[None,784])
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,w)+b)

#创建训练模型
y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))  # 计算交叉商=sum(  -y'*log(y) )
optimizer = tf.train.GradientDescentOptimizer(0.5) #学习率=0.5，根据差距进行反向传播修正参数
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer() # initialize_all_variables()  # 初始化训练结构
sess = tf.Session()                   # 建立Tensorflow训练会话
sess.run(init)                        # 将训练结构装载到会话中

# 采用mini_batch进行训练,mini_batch的大小为100
for i in range(50):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    # batch_xs_t = tf.convert_to_tensor(batch_xs)
    # batch_ys_t = tf.convert_to_tensor(batch_ys)
    sess.run(train, feed_dict={x: batch_xs, y: batch_ys}) # 训练train

# 评估模型
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  # 预测值和真实值之间是否相等，返回逻辑值
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))  # 将逻辑值转化为数值，并求平均值作为准确率

# 在测试集上计算正确率
print( sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})  )