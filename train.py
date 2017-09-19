# _*_ coding: utf-8 _*_
import RC_network
import load_data


import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DateFormatter 
import datetime
import pandas as pd
import os
from matplotlib.font_manager import FontProperties
from outliers import smirnov_grubbs as grubbs
import sys


font = FontProperties(fname=r"/Users/zhutq/mystuff/gdals/RC_model/simhei.ttf")


'''2种归一化的方式'''
def Min_Max_Normalization(array):
	N_array = []
	for x in array:
		x = float(x - np.min(array))/(np.max(array)- np.min(array))
		N_array.append(x)
	return N_array

def Min_Max_Normalization_2(array):
	N_array = []
	for x in array:
		#x = float(x - np.min(array))/(np.max(array)- np.min(array))
		x = float((0.8-0.2)*(x-np.min(array))/(np.max(array)- np.min(array)))+0.2
		N_array.append(x)
	return N_array

def Mean_Normalization(array):
	N_array = []
	for x in array:
		x = float(x - array.mean())/array.std()
		N_array.append(x)
	return N_array

'''使用格拉布斯准则剔除异常值
def grubbs_out(df):
	first_name = df.columns.values[0]
	print ('正在去除({})参数的异常值'.format(first_name))
	df_r = pd.DataFrame(grubbs.test(df[first_name],alpha=0.05))
	name_len = np.shape(df)[1]
	if name_len>1:
		name_list = df.columns.values
		for i in range(1,name_len):
			name = name_list[i]
			print ('正在去除({})参数的异常值'.format(name))
			df_i = pd.DataFrame(grubbs.test(df[name],alpha=0.05))
			df_r = df_r.join(df_i,how='outer')
	print ('丢弃所有的空值')
	return df_r.dropna()
'''

def grubbs_out(df,parameter):
	df_r = df[['记录时间']]
	for name in parameter:
			print ('正在去除({})参数的异常值'.format(name))
			df_i = pd.DataFrame(grubbs.test(df[name],alpha=0.05))
			df_r = df_r.join(df_i,how='outer')
	print ('丢弃所有的空值')
	return df_r.dropna()

def creat_parameter_list(df,parameter):
	result = df.copy()
	parameter_list=[]
	for p in parameter:
		name = result.filter(regex=p).columns.values[0]
		parameter_list.append(name)
	return parameter_list


def Pretreatment(df,parameter,p_type='min_max'):
	df_r = df.copy()
	#df_r = result[parameter]
	#df_r = grubbs_out(df_r)
	for p in parameter:
		values = df_r[p].values
		print ('正在对({})参数进行归一化'.format(p))
		if p_type == 'min_max':
			values = Min_Max_Normalization_2(values)
			#values = Min_Max_Normalization(values)
		elif p_type == 'mean':
			values = Mean_Normalization(values)
		elif p_type == 'None':
			pass 
		else:
			print ('归一化方式选择错误')
			sys.exit(0)
		df_r[p] = values
	#result_mat = result[parameter_list]
	return df_r
def select_hour(df,hour):
	dates = []
	hours = []
	for h in df['记录时间'].values:
		pass
	pass



def train_test_data(df,train_start_date='2012-05-01',train_end_date='2016-04-30',test_start_date='2016-05-01',test_end_date='2017-04-30'):
	train_start_date = datetime.datetime.strptime(train_start_date,'%Y-%m-%d')
	train_end_date = datetime.datetime.strptime(train_end_date,'%Y-%m-%d')
	test_start_date = datetime.datetime.strptime(test_start_date,'%Y-%m-%d')
	test_end_date = datetime.datetime.strptime(test_end_date,'%Y-%m-%d')

	mask_train = (df['记录时间'] > train_start_date) & (df['记录时间'] <= train_end_date)
	mask_test = (df['记录时间'] > test_start_date) & (df['记录时间'] <= test_end_date)
	print ('训练数据集时间为{}至{}'.format(train_start_date,train_end_date))
	df_train = df.loc[mask_train]
	print ('测试数据集时间为{}至{}'.format(test_start_date,test_end_date))
	df_test = df.loc[mask_test]
	return df_train,df_test


if __name__ == '__main__':
	'''
	训练步骤
	1.选定输入，输出的参数
	2.剔除异常值
	3.前4年的数据为训练数据，后1年的数据为测试数据
	4.归一化处理
	PH值	水温(℃)	溶解氧(mg/L)	浊度(NTU)	电导率(us/cm)	CODmn(mg/L)	氨氮(mg/L)	总磷(mg/L)	总氮(mg/L)	蓝绿藻(ug/L)	绿藻(ug/L)	硅甲藻(ug/L)	隐藻(ug/L)	叶绿素a(ug/L)

	'''
	excel_path = '/Users/zhutq/mystuff/gdals/dxsk.xls'
	 
	

	input_parameter = ['PH','溶解氧','总磷','总氮'] # ,'浊度','PH','电导率','COD'，'PH','电导率','COD','总氮'，'溶解氧','叶绿素a','总磷'
	output_parameter = ['PH','溶解氧','总磷','总氮']
	train_start_date = '2014-05-01'
	train_end_date = '2015-04-30'
	test_start_date = '2015-05-01'
	test_end_date = '2016-04-30'
	train_num = 100
	test_num = 10 
	dt = 0.1
	g= 1.5
	#p_type = 'None'
	#p_type='min_max'
	p_type = 'mean'


	save_dir = os.path.join('./result/'+output_parameter[0], '{}'.format(input_parameter)+'****'+p_type+'---'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
	log_path = os.path.join(save_dir+'/log.txt')
	if (not os.path.exists(save_dir)):
		print('创建文件夹{}'.format(save_dir))
		os.makedirs(save_dir)
	origin = sys.stdout 
	f = open(log_path, 'w') 
	sys.stdout = f 
	 
	if output_parameter[0] == 'standard':
		standard_flag = True
		WQ_name = input_parameter
	else:
		WQ_name = input_parameter + output_parameter
		standard_flag = False

	df = load_data.screen_excel(excel_path,WQ_name,na_flag='drop',standard=standard_flag)
	#df = 
	
	if standard_flag:
		input_parameter_list = creat_parameter_list(df,input_parameter)
		output_parameter_list = ['standard']
		WQ_parameter_list = input_parameter_list
	else:
		input_parameter_list = creat_parameter_list(df,input_parameter)
		output_parameter_list = creat_parameter_list(df,output_parameter)
		WQ_parameter_list = input_parameter_list +  output_parameter_list

	print ('**************************')
	print ('剔除异常值')
	grubbs_df = grubbs_out(df,WQ_parameter_list) #剔除异常值

	if standard_flag:
		grubbs_df = grubbs_df.join(df[['standard']],how='outer')
		grubbs_df.dropna()

	print ('**************************')
	print ('分割训练、测试数据')
	train_df,test_df = train_test_data(df=grubbs_df,train_start_date=train_start_date,
										train_end_date=train_end_date,
										test_start_date=test_start_date,
										test_end_date=test_end_date) #分段训练和测试数据
	print ('**************************')
	print ('对数据进行归一化处理，归一化方式为{}'.format(p_type))
	if (p_type != 'None'):
		train_df = Pretreatment(train_df,WQ_parameter_list,p_type) #归一化
		test_df = Pretreatment(test_df,WQ_parameter_list,p_type) 




	print ('------------------------')	
	print ('训练输入参数为{}'.format(input_parameter_list))
	train_input_data = np.mat(train_df[input_parameter_list].values.T)
	print ('训练输出参数为{}'.format(output_parameter_list))
	train_output_data = np.mat(train_df[output_parameter_list].values.T)

	

	print ('------------------------')
	print ('测试输入参数为{}'.format(input_parameter_list))
	test_input_data = np.mat(test_df[input_parameter_list].values.T)
	print ('测试输出参数为{}'.format(output_parameter_list))
	test_output_data = np.mat(test_df[output_parameter_list].values.T)

	print ('训练数据的大小为{}'.format(np.shape(train_input_data)))
	print ('测试数据的大小为{}'.format(np.shape(test_input_data)))

	num_input,num = np.shape(train_input_data)
	num_output = np.shape(train_output_data)[0]

	#print ('输入数据的维度为{}'.format(np.shape(input_data)))
	print ('训练数据每组数据含有{}个数据，共{}组'.format(num_input,num))
	
	print ("设定的g值={}".format(g))
	
	network = RC_network.network(num_input=num_input,num_output=num_output,g=g,dt=dt)
	RC = RC_network.Reservoir_Computing(input_data=train_input_data,output_data=train_output_data,network=network,save_dir=save_dir)
	RC.updata_testdata(test_input_data,test_output_data)
	RC.training(num_of_train=train_num,test_flag=True,time_num=test_num)
	
	sys.stdout = origin 
	f.close()