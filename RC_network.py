# _*_ coding: utf-8 _*_

#%matplotlib inline
import numpy as np 
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DateFormatter

from numpy.random import random_integers, randint, permutation
from scipy import rand, randn, ones, array
from scipy.sparse import csr_matrix,coo_matrix
from matplotlib.font_manager import FontProperties

class network(object):
	"""状态池网络结构的类
	todo：
	x  : 活性
	M  : 以0为标准差，1为方差的高斯分布的稀疏矩阵 
	wi : 输入数据的调整权重
	wo : 递归神经网络与输出的连接权重
	dt : 步长
	p  : 稀疏连接的概率
	wf : 系统生成的全0数据作为输入数据的叠加时的权重
	r  : x使用激活函数(tanh)归一化后的值
	z  : 输出
	g  : # 0.5，1.0，1.5，2
	alpha : #
	函数的含义

	todo:

	"""
	def __init__(self,num_input , num_output , num_node=1000 , g=1.5 , p=0.1 , alpha=1.5 , dt=0.1):
		super(network, self).__init__()
		self.num_input = num_input
		self.num_output = num_output
		self.N = num_node
		self.g = g
		self.p = p
		self.alpha = alpha
		self.dt = dt

		self.P = self.creat_P() # 大P
		self.x = self.creat_x() # x
		self.M = self.creat_M() # 大M
		self.wi = self.creat_wi() # wi
		self.wo = self.creat_wo() # wo
		self.wf = self.creat_wf() # wf
		self.r =  self.creat_r() # r
		self.z = self.creat_z() # z

	def _rand_sparse(self,m, n, density):
		# density 参数需要在 [0,1]范围内
		if density > 1.0 or density < 0.0:
			raise ValueError('density should be between 0 and 1')
		#这个函数可以增加非0元素的数量，也可以直接用scipy的sparse来生成
		nnz = max( min( int(m*n*density), m*n), 0)
		rand_seq = permutation(m*n)[:nnz]
		row  = rand_seq / n
		col  = rand_seq % n
		data = ones(nnz, dtype='int8')

		return coo_matrix( (data,(row,col)), shape=(m,n) )

	def sprand(self,m, n, density):
		"""Build a sparse uniformly distributed random matrix

		   Parameters
		   ----------

		   m, n     : dimensions of the result (rows, columns)
		   density  : fraction of nonzero entries.


		   Example
		   -------

		   >>> print sprand(2, 3, 0.5).todense()
		   matrix[[ 0.5724829   0.          0.92891214]
				 [ 0.          0.07712993  0.        ]]

		"""
		A = self._rand_sparse(m, n, density)
		A.data = rand(A.nnz)
		return A

	def sprandn(self,m, n, density):
		"""Build a sparse normally distributed random matrix

		   Parameters
		   ----------

		   m, n     : dimensions of the result (rows, columns)
		   density  : fraction of nonzero entries.


		   Example
		   -------

		   >>> print sprandn(2, 4, 0.5).todense()
		   matrix([[-0.84041995,  0.        ,  0.        , -0.22398594],
				   [-0.664707  ,  0.        ,  0.        , -0.06084135]])
		"""
		A = self._rand_sparse(m, n, density)
		A.data = randn(A.nnz)
		return A

	def creat_M(self):
		N = self.N
		g= self.g
		p = self.p
		scale = 1.0/np.sqrt(p*N)
		M = (self.sprandn(N,N,p).toarray())*g*scale
		return np.mat(M)

	def creat_wi(self):
		N = self.N
		num_input = self.num_input
		wi = randn(N,num_input)*0.5 #改成稀疏连接
		return np.mat(wi)

	def creat_wo(self):
		nRec2Out = self.N
		wo = (np.zeros([nRec2Out,self.num_output]))
		return np.mat(wo)

	def creat_dw(self):
		nRec2Out = self.N
		dw = (np.zeros([nRec2Out,1]))
		return np.mat(dw)

	def creat_wf(self):
		N = self.N
		num_output = self.num_output
		wf = 2.0 * (np.random.rand(N,num_output)-0.5)
		return np.mat(wf)

	def creat_P(self):
		nRec2Out = self.N
		alpha = self.alpha
		P = (1.0/alpha)*np.eye(nRec2Out)
		return np.mat(P)

	def creat_x(self):
		N = self.N
		x = 0.01*randn(N,1)
		return np.mat(x)

	def creat_r(self):
		x = self.x
		r = np.tanh(x)
		return np.mat(r)

	def creat_z(self):
		num_output = self.num_output
		z = 0.5*randn(num_output,1)
		return np.mat(z)


class Reservoir_Computing(object):
	"""
	这是一个状态池网络RC训练、测试的类

	输入数据:input_data(np.array)
	输出数据:output_data(np.array)
	网络结构:network(class network)
	学习率:learn_every(float)
	
	"""
	def __init__(self, input_data,output_data,network,save_dir=None,learn_every=1):
		super(Reservoir_Computing, self).__init__()
		self.input_data = np.mat(input_data)
		self.output_data = np.mat(output_data)
		self.network = network
		self.learn_every = learn_every		
		self.save_dir = save_dir

	def updata_testdata(self,test_in_data,test_out_data):
		self.test_in_data = test_in_data
		self.test_out_data = test_out_data
	
	def _update(self,P,r,k,z,test,ti,wo):
		k = np.dot(P,r)
		rPr = np.dot(np.transpose(r),k)
		c = 1.0/(1.0 + rPr)
		P = P - np.dot(k,(np.array(k).T*np.array(c)))

		'''更新error'''
		e = z - np.mat(test)[:,ti-1] #todo

		dw = np.mat(-np.array(e) * np.array(k).T*np.array(c))  #todo 
		wo = wo + dw.T #todo

	def _display_error_data(self,error_data,title,save_dir=None):
		plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
		plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
		plt.figure(figsize=(8, 6), dpi=400)
		plt.rcParams.update({'font.size': 25})
		plt.title(title)
		#plt.grid(True)

		x = range(1,len(error_data)+1)
		y = error_data

		plt.xlabel("迭代轮次")
		plt.ylabel("误差值")
		plt.xticks(fontsize=25)
		plt.yticks(fontsize=25)
		#plt.ylim(-2.0, 5.0)
		
		plt.plot(x,y, "g-", linewidth=2.0)#, label=title)
		

		plt.legend(loc="upper left", shadow=True)

		if save_dir != None:
			if (not os.path.exists(save_dir)):
				print('创建文件夹{}'.format(save_dir))
				os.mkdir(save_dir)
			path = save_dir+'/'+title
			print ('正在保存图片至{}'.format(path))
			plt.savefig(path)


		plt.show()

	def training(self,num_of_train=1000,test_flag=False,time_num=200):
		"""
		矩阵为np.array时，
		matlab :'.*'== numpy:'*'
		matlab :'*' == numpy :'np.dot()'

		矩阵为np.matrix时，
		matlab :'.*'== numpy:'np.multiply()'
		matlab :'*' == numpy :'*'

		"""
		x = self.network.x
		M = self.network.M
		wi = self.network.wi
		wo = self.network.wo
		dt = self.network.dt

		P = self.network.P
		wf = self.network.wf
		r = self.network.r
		z = self.network.z

		num_output = self.network.num_output
		
		y = self.input_data
		test = self.output_data

		_, data_len = np.shape(y)
		data_range = range(data_len)

		zt = np.mat(np.zeros(np.shape(test)))
		wo_len = np.zeros([1,data_len])

		self.error_data_avg =[]
		self.test_error_avg =[]
		for time in range(num_of_train):
			self.zt = zt
			if 0 == (time % time_num):
				print ('正在进行第{}次循环，拟合的结果和目标数据如下：'.format(time+1))
				train_title = 'The-{}-time-train'.format(time+1)
				#print ('zt: {}'.format(np.shape(zt)))
				#print ('test: {}'.format(np.shape(test)))
				#print ('data_range: {}'.format(np.shape(data_range)))
				self.display(np.array(self.zt)[0],np.array(test)[0],data_range,title=train_title,save_dir=self.save_dir)
				print ('开始训练')
			ti = 0
			for t in data_range:
				ti = ti + 1
				#if ti > 14435:
				#	print ('ti={}'.format(ti))
				if 0 == (ti % data_len/19):
					#print ('time:{}'.format(t))
					'''
					plt.figure(figsize=(8, 6), dpi=80)
					plt.xlabel('time(frames)')
					plt.ylabel('|w|')
					plt.plot(data_range,wo_len[0])
					plt.show()
					'''
				'''更新权重
				TODO:修改交换连接的wf，M。 交换其中N个非0与0的位置
				'''
				x = (1.0-dt)*x + np.dot(M,(r*dt)) + np.dot(wf,(z*dt)) + np.dot(wi,(np.mat(y)[:,ti-1])*dt) #todo
				r = np.tanh(x)
				z = np.dot(np.transpose(wo),r)


				#for i in range(10):
				if 0 == (ti % self.learn_every):
					k = np.dot(P,r)
					rPr = np.dot(np.transpose(r),k)
					c = 1.0/(1.0 + rPr)
					P = P - np.dot(k,(np.array(k).T*np.array(c)))

					'''更新error'''
					e = z - np.mat(test)[:,ti-1] #todo

					dw = np.mat(-np.array(e) * np.array(k).T*np.array(c))  #todo 
					wo = wo + dw.T #todo
				
				'''更新参数'''
				self.wo = wo
				np.mat(zt)[:,ti-1] = z #todo
				np.mat(wo_len)[:,ti-1] = np.sqrt(np.dot(wo.T,wo)) #todo
				#self.zt = zt
				self.wo_len = wo_len

			'''error值'''
			diff = np.mat(np.abs(zt - test))
			diff_len = np.shape(diff)[0]
			if diff_len == 1:
				error = diff
				error_avg = np.sum(error)/data_len
			else:
				diff_sum = np.multiply(diff[0,:],diff[0,:])
				for i in range(1,diff_len):
					diff_num = diff[i,:]
					diff_sum = diff_sum + np.multiply(diff_num,diff_num)
				error = np.sqrt(diff_sum)
				error_avg = np.sum(error)/data_len 
			self.error = error
			self.error_avg = error_avg

			self.error_data_avg.append(self.error_avg)

			'''保存网络结构'''
			self.network.x=x
			self.network.M=M
			self.network.wi=wi
			self.network.wo=wo
			self.network.dt=dt
			self.network.P=P
			self.network.wf=wf
			self.network.r=r
			self.network.z=z

			test_display_flag = False
			if (test_flag == True):
				
				if (0 == (time % time_num)):
					print ('训练至第{}轮，开始测试.....'.format(time+1))
					test_display_flag = True
					self.time = time
				#print("进行第{}次测试".format(time+1))
				zpt,test_error,test_error_avg = self.testing(self.test_in_data,self.test_out_data,self.network,test_display_flag)
				self.test_error_avg.append(test_error_avg)

		print ('Training MAE: {}'.format(error_avg))



		self.zt = zt
		self.zpt = zpt
		zt = np.array(zt)
		test = np.array(test)
		'''
		print('训练的最终效果如下：')
		self.display(zt[0],test[0],data_range,title='Train-result',save_dir=self.save_dir)

		print ("训练的误差走势图：")
		self._display_error_data(self.error_data_avg,"Training Error",save_dir=self.save_dir)
		if self.test_error_avg:
			print ("测试的误差走势图：")
			self._display_error_data(self.test_error_avg,"Testing Error",save_dir=self.save_dir)
		'''

		

	def testing(self,test_in_data,test_out_data,network,test_display_flag=False):
		x = self.network.x
		M = self.network.M
		wi = self.network.wi
		wo = self.network.wo
		dt = self.network.dt

		P = self.network.P
		wf = self.network.wf
		r = self.network.r
		z = self.network.z

		
		num_output = self.network.num_output

		y = test_in_data
		test = test_out_data
		_, data_len = np.shape(y)
		data_range = range(data_len)

		zpt = np.zeros(np.shape(test))
		wo_len = np.zeros([1,data_len])

		
		ti = 0
		for t in data_range:
			ti = ti + 1
			
			x = (1.0-dt)*x + np.dot(M,(r*dt)) + np.dot(wf,(z*dt)) + np.dot(wi,(np.mat(y)[:,ti-1])*dt) #todo
			r = np.tanh(x)
			z = np.dot(np.transpose(wo),r)

			np.mat(zpt)[:,ti-1] = z 
			np.mat(wo_len)[:,ti-1] = np.sqrt(np.dot(wo.T,wo))

		diff = np.mat(np.abs(zpt - test))
		diff_len = np.shape(diff)[0]
		if diff_len == 1:
			error = diff
			error_avg = np.sum(error)/data_len
		else:
			diff_sum = np.multiply(diff[0,:],diff[0,:])
			for i in range(1,diff_len):
				diff_num = diff[i,:]
				diff_sum = diff_sum + np.multiply(diff_num,diff_num)
			error = np.sqrt(diff_sum)
			error_avg = np.sum(error)/data_len 
		test_error = error
		test_error_avg = error_avg
		if test_display_flag == True:
			print ('测试的效果如下')
			self.display(np.array(zpt)[0],np.array(test)[0],data_range,output_label='target_data',title='Test-result-'+str(self.time),save_dir=self.save_dir)

			print ('Testing MAE: {}'.format(test_error_avg))
		return zpt,test_error,test_error_avg


	def display(self,input_data,output_data,time_data,output_label='data',title='RC_network',save_dir=None):
		font = FontProperties(fname=r"/Users/zhutq/mystuff/gdals/RC_model/simhei.ttf")
		colorstyle = ['r','b','g','tan','k','y','m']
		''' matplotlib 画图函数

			Input:
				input_data (np.array)  : 输入函数
				output_data   (np.array)  : 目标函数
				time_data (np.array): 时间轴
			Output:
				图画
		'''

		x = time_data
		y0 = input_data
		y1 = output_data

		# 生成画布
		plt.figure(figsize=(16, 12), dpi=180)
		plt.title(title,fontproperties=font)
		plt.grid(True)

		plt.xlabel("Time",fontproperties=font)
		#plt.xlim(-200 , 1600)

		plt.ylabel("Data",fontproperties=font)
		#plt.ylim(-2.0, 2.0)
		plt.plot(x,y1, "g-", linewidth=2.0, label=output_label)
		plt.plot(x,y0, "r-", linewidth=2.0, label="simulation")

		plt.legend(loc="upper left", shadow=True)

		if save_dir != None:
			if (not os.path.exists(save_dir)):
				print('创建文件夹{}'.format(save_dir))
				os.mkdir(save_dir)
			path = save_dir+'/'+title
			print ('正在保存图片至{}'.format(path))
			plt.savefig(path)


		plt.show()


if __name__ == '__main__':
	N = 1000
	g = 1.5
	p = 0.1
	alpha = 1.5
	dt = 0.1

	learn_every = 2
	nsecs = 1440
	nRec2Out = N
	amp = 1.3
	freq = 1.0/60

	pi = np.pi
	sin = np.sin

	simtime = np.linspace(0,nsecs,int(nsecs/dt),endpoint=False)
	simtime_len = len(simtime)

	ft = (amp/1.0)*sin(1.0*pi*freq*simtime) + \
			(amp/2.0)*sin(2.0*pi*freq*simtime) + \
			(amp/6.0)*sin(3.0*pi*freq*simtime) + \
			(amp/3.0)*sin(4.0*pi*freq*simtime)
	ft = ft/1.5


	ft2 = (amp/1.0)*sin(1.0*pi*freq*simtime) + \
			(amp/2.0)*sin(2.0*pi*freq*simtime) + \
			(amp/6.0)*sin(3.0*pi*freq*simtime) + \
			(amp/3.0)*sin(4.0*pi*freq*simtime)
	ft2 = ft2/1.5

	zt = (np.zeros([1,simtime_len]))

	'''
	输入与输出的格式需要理清楚，主要目的是不混淆training中的参数更新结构。
	输入：M * N
	输出：m * n 
	参数：？* ？

	'''

	input_data = np.mat(zt)
	output_data = np.mat(ft)
	test_in_data = np.mat(zt)
	test_out_data = np.mat(ft2)
	num_input = np.shape(input_data)[0]
	num_output = np.shape(np.mat(output_data))[0]



	
	network =network(num_input=num_input,num_output=num_output,num_node=N,p=p,g=g,alpha=alpha,dt=dt)
	RC_network = Reservoir_Computing(input_data=input_data,output_data=output_data,network=network,learn_every=learn_every)

	#RC_network.display(zt[0],ft,simtime)
	RC_network.updata_testdata(test_in_data,test_out_data)

	RC_network.training(num_of_train=1,test_flag=True)
	#RC_network.save_data()
	#test_result,test_error,test_error_avg = RC_network.testing(test_in_data,test_out_data,RC_network.network)
	



