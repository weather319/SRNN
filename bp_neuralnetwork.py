# -*- coding: utf-8 -*-
#import tensorflow.contrib.learn as skflow


"""
神经网络设计：选用3层BP神经网络。
输入层为近红外、红光、绿光、短波红外数据。
隐含层传递函数为双曲S型正切函数“tansig”。
输出层采用的是线性函数“purelin”来拟合水质参数。
水质参数选择：
"""
import numpy as np  
  
  

  
#定义NeuralNetwork 神经网络算法  
class NeuralNetwork:  
	#初始化，layes表示的是一个list，eg[10,10,3]表示第一层10个神经元，第二层10个神经元，第三层3个神经元
	def __init__(self, layers, activation='tanh'):  
		""" 
		:param layers: layes表示的是一个list,包含了每一层的神经元数目. 
		至少包括2层神经网络，也就是list至少有2个数。 
		:param activation: 激活函数 "logistic" 或 "tanh" 
		"""  
		if activation == 'logistic':  
			self.activation = self.logistic  
			self.activation_deriv = self.logistic_derivative  
		elif activation == 'tanh':  
			self.activation = self.tanh  
			self.activation_deriv = self.tanh_deriv  
		# 存储权值矩阵
		self.weights = []  
		#循环从1开始，相当于以第二层为基准，进行权重的初始化  
		for i in range(1, len(layers) - 1):  
			#对当前神经节点的前驱赋值  
			self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i] + 1))-1)*0.25)  
			#对当前神经节点的后继赋值  
			self.weights.append((2*np.random.random((layers[i] + 1, layers[i + 1]))-1)*0.25)  

	#定义双曲函数和他们的导数
	def tanh(self,x):  
		return np.tanh(x)  

	def tanh_deriv(self,x):  
		return 1.0 - np.tanh(x)**2  

	def logistic(self,x):  
		return 1/(1 + np.exp(-x))  

	def logistic_derivative(self,x):  
		return logistic(x)*(1-logistic(x))    
	#训练函数   ，X矩阵，每行是一个实例 ，y是每个实例对应的结果，learning_rate 学习率，   
	# epochs，表示抽样的方法对神经网络进行更新的最大次数  
	def fit(self, x, y, learning_rate=0.05, epochs=10000):  
		#print ('正在进行训练')
		X = np.atleast_2d(x) #确定X至少是二维的数据  
		temp = np.ones([X.shape[0], X.shape[1]+1]) #初始化矩阵  
		temp[:, 0:-1] = X  # adding the bias unit to the input layer  
		X = temp  
		y = np.array(y) #把list转换成array的形式  

		for k in range(epochs):
			#if k%1000 == 0:
			#	print ('迭代第{}次'.format(k))  
			#随机选取一行，对神经网络进行更新  
			i = np.random.randint(X.shape[0])   
			a = [X[i]]  

			#完成所有正向的更新  
			for l in range(len(self.weights)):
				dot_value = np.dot(a[l], self.weights[l])   # 权值矩阵中每一列代表该层中的一个结点与上一层所有结点之间的权值
				activation = self.activation(dot_value)
				a.append(activation)  
			#  
			error = y[i] - a[-1]  
			deltas = [error * self.activation_deriv(a[-1])]  

			#开始反向计算误差，更新权重  
			for l in range(len(a) - 2, 0, -1): # we need to begin at the second to last layer  
				deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))  
			deltas.reverse()  
			for i in range(len(self.weights)):  
				layer = np.atleast_2d(a[i])  
				delta = np.atleast_2d(deltas[i])  
				self.weights[i] += learning_rate * layer.T.dot(delta)  

	#预测函数              
	def predict(self, x):  
		x = np.array(x)  
		temp = np.ones(x.shape[0]+1)  
		temp[0:-1] = x  
		a = temp  # a为输入向量(行向量)
		for l in range(0, len(self.weights)):  
			a = self.activation(np.dot(a, self.weights[l]))  
		return a 

	def test(self,X,Y,error):
		count = 0
		number = len(Y)
		print ('设定可接受的数据误差为{}'.format(error))
		for i in range(number):
			result = self.predict(X[i])
			print ('预测数据为{}，实际数据为{}'.format(result,Y[i]))
			if abs((Y[i]-result).sum()) < error:
				count = count + 1
		Correct_rate = (count/number)*100
		print ('测试正确的数量为{}条，测试样本为{}条'.format(count,number))
		print ('测试正确率为[{}%]'.format(Correct_rate))
		






if __name__ == '__main__':
	nn = NeuralNetwork(layers=[4,3,2])     # 网络结构: 2输入1输出,1个隐含层(包含2个结点)

	X = np.array([[0, 0 ,0 ,0 ],           # 输入矩阵(每行代表一个样本,每列代表一个特征)
                  [0, 1, 0 ,1 ],
                  [1, 0, 1, 0],
                  [1, 1, 1, 1]])

	#Y = np.array([[0, 1, 1, 1],
	#			[0,1,1,1]])
	
	Y = np.array([[0,0],
				  [1,1],
				  [1,1],
				  [1,1]])
	nn.fit(X, Y)                    # 训练网络 

	nn.test(X,Y,0.1)
	#print ('w:', nn.weights)          # 调整后的权值列表

	






























