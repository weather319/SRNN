'''
本程序为训练和预测TH水质数据趋势
输入：单一数据的历史数据
输出： 往后一个月的水质数据
预测步骤为：设定水质预测区间
'''

import RC_network
import load_data
from bp_neuralnetwork import NeuralNetwork

from train import grubbs_out,creat_parameter_list,Pretreatment,train_test_data
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DateFormatter 
import datetime
import pandas as pd
import os
import sys


class TH_network(RC_network.Reservoir_Computing):
	"""docstring for TH_network"""

	def drop_out(self,mat,percent):
		'''设定drop_out的概率得到需要更换的神经元个数
		随机选取wf中的非0数值位置，使其变为0
		
		example:
		input : 0.01
		num = 1000*0.01 = 10个
		需要调整10个神经元
		'''
		#zero_index = np.where(mat==0)
		#zero_x = np.squeeze(np.asarray(zero_index[0]))
		#zero_y = np.squeeze(np.asarray(zero_index[1]))
		connect_index = np.where(mat!=0)
		connect_x = np.squeeze(np.asarray(connect_index[0]))
		connect_y = np.squeeze(np.asarray(connect_index[1]))

		num = int(np.shape(mat)[0]*percent)
		for i in range(num):
			#zero_random_x = np.random.choice(zero_x)
			#zero_random_y = np.random.choice(zero_y)
			conncet_random_x = np.random.choice(connect_x)
			conncet_random_y = np.random.choice(connect_y)
			#conncet_value = mat[conncet_random_x,conncet_random_y]
			mat[conncet_random_x,conncet_random_y] = 0.0
			#mat[zero_random_x,zero_random_y] = conncet_value
		return mat

	def update_WQ_parameters(self,WQ_parameter):
		self.WQ_parameter = WQ_parameter
	
	def display(self,input_data,output_data,time_data,output_label='data',title='RC_network',save_name='',save_dir=None):
		#font = FontProperties(fname=r"/Users/zhutq/mystuff/gdals/RC_model/simhei.ttf")
		plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
		plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
		plt.rcParams.update({'font.size': 15})
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
		plt.figure(figsize=(8, 6), dpi=600)
		plt.xticks(fontsize=25)
		plt.yticks(fontsize=25)
		plt.title(title)
		#plt.grid(True)

		plt.xlabel("时间（天）")
		#plt.xlim(-200 , 1600)

		plt.ylabel("水质指标值")
		plt.ylim(0, 1.0)
		plt.plot(x,y1, "g-", linestyle= '-',linewidth=1.0, label='水体指标实际值')
		plt.plot(x,y0, "r-", linestyle= '--', linewidth=1.0, label="网络输出预测值")

		plt.legend(loc="upper left") #, shadow=True

		if save_dir != None:
			if (not os.path.exists(save_dir)):
				print('创建文件夹{}'.format(save_dir))
				os.mkdir(save_dir)
			path = save_dir+'/'+save_name+'.svg'
			print ('正在保存图片至{}'.format(path))
			plt.savefig(path)


		#plt.show()


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
		#wo_len = np.zeros([1,data_len])

		self.error_data_avg =[]
		self.test_error_avg =[]
		self.test_error =[]
		print ('开始训练')
		for time in range(num_of_train):
			#wf = self.drop_out(wf,0.01)
			self.zt = zt
			if 0 == (time+1 % 100):
				print ('正在进行第{}次循环，拟合的结果和目标数据如下：'.format(time+1))
				for i in range(num_output):
					wq = self.WQ_parameter[i]
					train_title = '{}水质参数的第{}次训练结果'.format(wq,(time+1))
					self.display(np.array(self.zt)[i],np.array(test)[i],data_range,output_label=wq,title='{}水体指标训练结果'.format(wq),save_name=train_title,save_dir=self.save_dir)
			ti = 0
			for t in data_range:
				ti = ti + 1
				'''更新权重'''
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
				#np.mat(wo_len)[:,ti-1] = np.sqrt(np.dot(wo.T,wo)) #todo
				#self.zt = zt
				#self.wo_len = wo_len

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
				if (0 == (time+1 % time_num)):
					print ('训练至第{}轮，开始测试.....'.format(time+1))
					test_display_flag = True
					self.time = time
				self.zpt,test_error,test_error_avg = self.testing(self.test_in_data,self.test_out_data,self.network,test_display_flag)
				self.test_error_avg.append(test_error_avg)
				self.test_error.append(test_error)
				#print("进行第{}次测试".format(time+1))

				

		print ('Training MAE: {}'.format(error_avg))



		self.zt = zt
		self.test_error = np.array(self.test_error).T
		'''
		print('训练的最终效果如下：')
		for i in range(num_output):
			wq = self.WQ_parameter[i]
			train_title = '{}水质参数的最终训练结果'.format(wq)
			self.display(np.array(self.zt)[i],np.array(test)[i],data_range,output_label=wq,title='{}水体指标训练结果'.format(wq),save_name=train_title,save_dir=self.save_dir)
	
		#self.display(zt[0],test[0],data_range,title='Train-result',save_dir=self.save_dir)

		print ("训练的误差走势图：")
		self._display_error_data(self.error_data_avg,"训练误差图",save_dir=self.save_dir)
		print ('训练误差如下:')
		print (self.error_data_avg)
		print 
		if (test_flag == True):
			print ("测试的误差走势图：")
			self._display_error_data(self.test_error_avg,"测试误差图",save_dir=self.save_dir)
			print ('测试误差如下:')
			print (self.test_error_avg)
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
		#num_output,_ = np.shape(test)
		data_range = range(data_len)

		zpt = np.zeros(np.shape(test))
		#wo_len = np.zeros([1,data_len])

		
		ti = 0
		test_correct_error = 0
		for t in data_range:

			ti = ti + 1
			
			x = (1.0-dt)*x + np.dot(M,(r*dt)) + np.dot(wf,(z*dt)) + np.dot(wi,(np.mat(y)[:,ti-1])*dt) #todo
			r = np.tanh(x)
			z = np.dot(np.transpose(wo),r)
			self.z = z

			np.mat(zpt)[:,ti-1] = z
			_error = abs(np.array(test[:,ti-1]) - np.array(z)) 

			#np.mat(wo_len)[:,ti-1] = np.sqrt(np.dot(wo.T,wo))
			test_correct_error += np.sqrt(_error*_error)
		#self.test_error_avg.append(test_correct_error/data_len)
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
			for i in range(num_output):
				wq = self.WQ_parameter[i]
				title = '{}参数第{}次测试结果'.format(wq,str(self.time+1))
				self.display(np.array(zpt)[i],np.array(test)[i],data_range,output_label=wq,title='{}水体指标测试结果'.format(wq),save_name=title,save_dir=self.save_dir)

			print ('Testing MAE: {}'.format(test_error_avg))
		return zpt,test_correct_error/data_len,test_error_avg

def creat_parameter_list(df,parameter):
	result = df.copy()
	parameter_list=[]
	for p in parameter:
		name = result.filter(regex=p).columns.values[0]
		parameter_list.append(name)
	return parameter_list

#def load_data(path,p_lists):
#	pass

def choice_by_day(df,time='0:00'):
	'''
	选取每天的12点水质数据为内容，若缺失12点的数据，则跳过该天
	begin = datetime.date(2014,6,1)  
    end = datetime.date(2014,6,7)  
    for i in range((end - begin).days+1):  
        day = begin + datetime.timedelta(days=i)  
        print str(day)
	'''
	df_r = df.copy()
	df_r.index = df_r['记录时间'].tolist()
	day_df = df_r.at_time(time)
	return day_df
class bp_train(object):
	"""bp_train 是使用bp神经网络预测水质的一种方法"""
	def __init__(self, input_data,output_data,network,save_dir=None):
		super(bp_train, self).__init__()
		#self.input_data = np.mat(input_data)
		#self.output_data = np.mat(output_data)
		#self.output_data_conv = self.conv_standard(self.output_data).T
		self.train_X = np.array(input_data).T
		self.train_Y = np.array(output_data).T

		self.network = network	
		self.save_dir = save_dir

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

	def display(self,input_data,output_data,time_data,output_label='data',title='RC_network',save_name='',save_dir=None):
		#font = FontProperties(fname=r"/Users/zhutq/mystuff/gdals/RC_model/simhei.ttf")
		plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
		plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
		plt.rcParams.update({'font.size': 15})
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
		plt.figure(figsize=(8, 6), dpi=180)
		plt.xticks(fontsize=25)
		plt.yticks(fontsize=25)
		plt.title(title)
		#plt.grid(True)

		plt.xlabel("时间（天）")
		#plt.xlim(-200 , 1600)

		plt.ylabel("水质指标值")
		plt.ylim(0, 1.0)
		plt.plot(x,y1, "g-", linestyle= '-',linewidth=1.0, label='水体指标实际值')
		plt.plot(x,y0, "r-", linestyle= '--', linewidth=1.0, label="网络输出预测值")

		plt.legend(loc="upper left") #, shadow=True

		if save_dir != None:
			if (not os.path.exists(save_dir)):
				print('创建文件夹{}'.format(save_dir))
				os.mkdir(save_dir)
			path = save_dir+'/'+save_name
			print ('正在保存图片至{}'.format(path))
			plt.savefig(path)


		plt.show()
	def updata_testdata(self,test_in_data,test_out_data):
		self.test_X = np.array(test_in_data).T
		#self.test_out_data = test_out_data
		self.test_Y = np.array(test_out_data).T

	def update_WQ_parameters(self,WQ_parameter):
		self.WQ_parameter = WQ_parameter


	def training(self,num_of_train=1000,test_flag=False,time_num=200):
		
		#train_output = np.array(self.output_data.T)
		#test_output = np.array(self.test_out_data.T)

		train_data_len,input_num = np.shape(self.train_X)
		_,num_output = np.shape(self.train_Y)
		test_data_len,_ = np.shape(self.test_X)
		train_result = np.zeros(np.shape(self.train_Y))
		test_result = np.zeros(np.shape(self.test_Y))
		self.train_error_avg =[]
		self.test_error_avg =[]

		for time in range(num_of_train):
			self.network.fit(self.train_X,self.train_Y,epochs=1)
			train_correct_error = 0.
			for j in range(train_data_len):
				result = self.network.predict(self.train_X[j])
				train_result[j] = result
				_error = abs(result - self.train_Y[j])#.sum()
				train_correct_error += np.sqrt(_error*_error)
			self.train_error_avg.append(train_correct_error/train_data_len)	
			if (time+1)%200 == 0:
				data_range,_ = np.shape(self.train_Y)
				data_range = range(data_range)
				print ('正在进行第{}次循环，拟合的结果和目标数据如下：'.format(time+1))
				for i in range(num_output):
					wq = self.WQ_parameter[i]
					train_title = '{}水质参数的第{}次训练结果'.format(wq,(time+1))
					self.display(train_result.T[i],self.train_Y.T[i],data_range,output_label=wq,title='{}水体指标训练结果'.format(wq),save_name=train_title,save_dir=self.save_dir)
				#print ('正在进行第{}次训练'.format(time+1))			
			
			#print ('第{}次训练的正确率为{}'.format(time+1,train_acc))
			
			if test_flag == True:
				#print ('训练至第{}轮，开始测试.....'.format(time+1))
				data_range,_ = np.shape(self.test_Y)
				data_range = range(data_range)
				test_correct_error = 0.
				for k in range(test_data_len):
					result = self.network.predict(self.test_X[k])
					#result = self.deconv_standard(output)
					test_result[k] = result
					_error = abs(result - self.test_Y[k])#.sum()
					#_error = self.standard_error(diff)
					test_correct_error += np.sqrt(_error*_error)
				self.test_error_avg.append(test_correct_error/test_data_len)
				if (time+1)%time_num ==0:
					for i in range(num_output):
						wq = self.WQ_parameter[i]
						test_title = '{}水质参数的第{}次测试结果'.format(wq,(time+1))
						self.display(test_result.T[i],self.test_Y.T[i],data_range,output_label=wq,title='{}水体指标测试结果'.format(wq),save_name=test_title,save_dir=self.save_dir)
				#print ('第{}次训练的正确率为{}'.format(time+1,test_acc))
				
				#print ("*" * 20 + "第{}次测试正确率: {}".format(time+1,test_acc) + " | 最佳正确率: {}".format(best_result))
				#print ("*" * 20 + "第{}次测试误差: {}".format(time+1,test_correct_error) )
				
		#print ("*" * 20 + '最佳正确率: {}'.format(best_result))
		self.train_result = train_result
		#print ('测试数据总数为{},最佳正确条数为'.format(self.test_total,self.best_correct))

		print ("训练的误差走势图：")
		self._display_error_data(self.train_error_avg,"训练误差",save_dir=self.save_dir)
		if self.test_error_avg:
			print ("测试的误差走势图：")
			self._display_error_data(self.test_error_avg,"测试误差",save_dir=self.save_dir)
		self.test_error = np.array(self.test_error_avg).T

if __name__ == '__main__':
#def main():
	'''
	训练步骤
	1.选定输入，输出的参数 
	2.剔除异常值
	3. 归一化处理
	4. 训练数据、测试数据分段
	PH值	水温(℃)	溶解氧(mg/L)	浊度(NTU)	电导率(us/cm)	CODmn(mg/L)	氨氮(mg/L)	总磷(mg/L)	总氮(mg/L)	蓝绿藻(ug/L)	绿藻(ug/L)	硅甲藻(ug/L)	隐藻(ug/L)	叶绿素a(ug/L)

	'''
	excel_path = 'your_excel'

	input_parameter = ['氨氮','溶解氧', 'CODmn', '总磷', '总氮'] # ,'浊度','PH','电导率','COD'，'PH','电导率','COD','总氮'，'溶解氧','叶绿素a','总磷'
	output_parameter = ['氨氮','溶解氧', 'CODmn', '总磷', '总氮']#,'CODmn', '总磷', '总氮']
	train_start_date = '2012-05-01'
	train_end_date = '2014-04-30'
	test_start_date = '2014-05-01'
	test_end_date = '2015-04-30'
	day_time = 7
	hour_time = '4:00'
	train_num = 15
	test_num = 1000 
	dt = 0.15
	g= 0.5
	alpha = 1.5
	#p_type = 'None'
	p_type='min_max'
	#p_type = 'mean'
	


	save_dir = os.path.join('./result/{}'.format(output_parameter), '{}'.format(input_parameter)+'****'+p_type+'---'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
	log_path = os.path.join(save_dir+'/log.txt')
	
	if (not os.path.exists(save_dir)):
		print('创建文件夹{}'.format(save_dir))
		os.makedirs(save_dir)
	
	origin = sys.stdout 
	f = open(log_path, 'w') 
	sys.stdout = f 
	
	print ('主要参数设定为dt={},g={},alpha={}'.format(dt,g,alpha))

	if output_parameter[0] == 'standard':
		standard_flag = True
		WQ_name = input_parameter
	else:
		WQ_name = input_parameter
		standard_flag = False

	df = load_data.screen_excel(excel_path,WQ_name,na_flag='fill',standard=standard_flag)
	#df = 
	
	if standard_flag:
		input_parameter_list = creat_parameter_list(df,input_parameter)
		output_parameter_list = ['standard']
		WQ_parameter_list = input_parameter_list
	else:
		input_parameter_list = creat_parameter_list(df,input_parameter)
		output_parameter_list = creat_parameter_list(df,output_parameter)
		WQ_parameter_list = input_parameter_list

	print ('**************************')
	print ('剔除异常值')
	grubbs_df = grubbs_out(df,WQ_parameter_list) #剔除异常值

	print ('**************************')
	print ('对数据进行归一化处理，归一化方式为{}'.format(p_type))

	if (p_type != 'None'):
		grubbs_df = Pretreatment(grubbs_df,WQ_parameter_list,p_type) #归一化

	if standard_flag:
		grubbs_df = grubbs_df.join(df[['standard']],how='outer')
		grubbs_df.dropna()



	print ('**************************')
	print ('分割训练、测试数据')
	train_df,test_df = train_test_data(df=grubbs_df,train_start_date=train_start_date,
										train_end_date=train_end_date,
										test_start_date=test_start_date,
										test_end_date=test_end_date) #分段训练和测试数据

	#if (p_type != 'None'):
		#train_df = Pretreatment(train_df,WQ_parameter_list,p_type) #归一化
		#test_df = Pretreatment(test_df,WQ_parameter_list,p_type) 

	print ('**************************')
	print ('选取每天的{}水质数据为内容，若缺失该时间数据，则跳过该天'.format(hour_time))

	train_df = choice_by_day(train_df,hour_time)
	test_df = choice_by_day(test_df,hour_time)

	print ('------------------------')	
	print ('训练输入参数为{}'.format(input_parameter_list))
	train_input_data = train_df[input_parameter_list].values
	print ('训练输出参数为{}'.format(output_parameter_list))
	train_output_data = train_df[output_parameter_list].values

	new_train_in = []
	new_train_out = []
	
	print ('------------------------')	
	print ('预测的时间间隔为{}'.format(day_time))
	for i in range(len(train_input_data)-day_time):
		_x = []
		_y = train_output_data[i+day_time]
		for j in range(day_time):
			_x += list(train_input_data[i+j])
		new_train_in.append(_x)
		new_train_out.append(_y)

	train_input_data = np.mat(new_train_in).T
	train_output_data = np.mat(new_train_out).T

	print ('训练输入数据的大小为{}'.format(np.shape(train_input_data)))
	print ('训练输出数据的大小为{}'.format(np.shape(train_output_data)))

	print ('------------------------')
	print ('测试输入参数为{}'.format(input_parameter_list))
	test_input_data = test_df[input_parameter_list].values
	print ('测试输出参数为{}'.format(output_parameter_list))
	test_output_data = test_df[output_parameter_list].values

	new_test_in = []
	new_test_out = []
	for i in range(len(test_input_data)-day_time):
		_x = []
		_y = test_output_data[i+day_time]
		for j in range(day_time):
			_x += list(test_input_data[i+j])
		new_test_in.append(_x)
		new_test_out.append(_y)

	test_input_data = np.mat(new_test_in).T
	test_output_data = np.mat(new_test_out).T


	print ('测试输入数据的大小为{}'.format(np.shape(test_input_data)))
	print ('测试输出数据的大小为{}'.format(np.shape(test_output_data)))


	num_input,num = np.shape(train_input_data)
	num_output,_ = np.shape(train_output_data)

	#print ('输入数据的维度为{}'.format(np.shape(input_data)))
	#print ('训练数据每组数据含有{}个数据，共{}组'.format(num_input,num))
	
	print ("设定的g值={}".format(g))
	'''
	network = RC_network.network(num_input=num_input,num_output=num_output,g=g,alpha=alpha,dt=dt)
	RC = TH_network(input_data=train_input_data,output_data=train_output_data,network=network,save_dir=save_dir)
	RC.updata_testdata(test_input_data,test_output_data)
	RC.update_WQ_parameters(WQ_name)
	RC.training(num_of_train=train_num,test_flag=True,time_num=test_num)
	'''
	
	
	network = NeuralNetwork(layers=[num_input,13,num_output])
	bp = bp_train(input_data=train_input_data,output_data=train_output_data,network=network,save_dir=save_dir)
	bp.updata_testdata(test_input_data,test_output_data)
	bp.update_WQ_parameters(WQ_name)
	bp.training(num_of_train=train_num,test_flag=True,time_num=test_num)
	
	sys.stdout = origin 
	f.close()
	
#if __name__ == '__main__':
	#alpha_lists = [1.0,1.1,1.2,1.3,1.4,1.5]
	#g_lists =[1.0,1.5]

	#main()