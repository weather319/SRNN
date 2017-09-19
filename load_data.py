# _*_ coding: utf-8 _*_
import pandas as pd
import numpy as np
import os
#import seaborn as sns
import matplotlib.pyplot as plt 
from matplotlib.font_manager import FontProperties
import datetime

from outliers import smirnov_grubbs as grubbs

def screen_excel(excel_path,WQ_name,na_flag='drop',standard=False):
	"""
	excel_path  为excel文件路径

	WQ_name   为要提取的水质参数名称，
	可为部分名称，但是不能与其他水质参数重复


	"""
	if os.path.exists(excel_path) and os.path.isfile(excel_path):
		print ('成功刚打开{}的水质表'.format(excel_path))
		df = pd.read_excel(excel_path)
	else:
		print ('打开错误，请检查水质表的文件路径')
		exit(0)
	dates = []
	hours = []
	for d in df['记录时间'].values:
		date = datetime.datetime.strptime(d,'%Y/%m/%d %H:%M:%S')
		hour = date.hour
		dates.append(date)
		#hours.append(hour)
	#df['记录时间'] = dates

	#df['hour'] = hours

	#df['记录时间'] = pd.to_datetime(df['记录时间'])# str to tiimeframe
	WaterQualiteId_lists = WQ_name
	WQId_name = [] #由于输入的id为简称，需要识别全称
	col_list = ['记录时间']
	print ('水质参数列表为 {}'.format(WaterQualiteId_lists))
	for WaterQualiteId in WaterQualiteId_lists:
		waterId_name = df.filter(regex=WaterQualiteId).columns.values[0]
		WQId_name.append(waterId_name)
		col_list.append(waterId_name)
	print ('col_list is {}'.format(col_list))
	print ('WQId_name is {}'.format(WQId_name))
	
	water_df = df.copy()
	'''内容为文本内容，变换为float'''
	water_df = water_df.replace(r'\s+', np.nan, regex=True)
	water_df = water_df.replace(r'无效数据', np.nan, regex=True)
	water_df = water_df.drop(['记录时间'],axis=1).astype(float)
	water_df['记录时间'] = dates
	#water_df[WQId_name] = water_df[WQId_name].astype(float)# 数据变为float形式
	water_df = water_df.reset_index(drop=True)
	'''对空数据操作'''
	if na_flag == 'drop':
		print ('对na数据进行丢弃操作')
		water_df = water_df.dropna()
	if na_flag == 'fill':
		print ('对na数据进行填充0操作')
		water_df = water_df.fillna(0)
	if standard == True:
		water_df = Quality_standard(water_df) #水质分级
		df_r = water_df[col_list].copy()
		df_r['standard'] = water_df['standard']
	else:
		df_r = water_df[col_list].copy()

	return df_r#,WQId_name

def df_plot(df):
	font = FontProperties(fname=r"simhei.ttf")
	ax = df.plot(figsize=(16,12))
	labels = ax.get_xticklabels()+ax.legend().texts+[ax.title]
	for label in labels : 
		label.set_fontproperties(font)
	plt.show()

def Quality_standard(df):
	'''根据地表水环境质量标准，确定水体的水质等级
	NH3_N 氨氮   0.15 0.50 1.0 1.50 2.0 
	DO 溶解氧    7.5 6.0 5.0 3.0 2.0
	COD 化学需氧量  ***缺少***
	CODmn 高锰酸钾指数. 2.0 4.0 6.0 10 15
	TP 总磷      0.02 0.10 0.20 0.30 0.40
	TN 总氮      0.20 0.50 1.0  1.5  2.0
	'''
	#lists = ['NH3_N','DO','CODmn','TP','TN']
	lists = ['氨氮(mg/L)','溶解氧(mg/L)','CODmn(mg/L)','总磷(mg/L)','总氮(mg/L)']   
	standard_list = []
	for i in df.index.values:
		if (df['氨氮(mg/L)'].loc[i] <= 0.15 and 
			df['溶解氧(mg/L)'].loc[i] >= 7.5 and 
			df['CODmn(mg/L)'].loc[i] <= 2.0 and 
			df['总磷(mg/L)'].loc[i] <= 0.02 and 
			df['总氮(mg/L)'].loc[i] <= 0.20):
			standard = 1
			standard_list.append(standard)
			continue
		elif (df['氨氮(mg/L)'].loc[i] <= 0.5 and 
			df['溶解氧(mg/L)'].loc[i] >= 6.0 and 
			df['CODmn(mg/L)'].loc[i] <= 4.0 and 
			df['总磷(mg/L)'].loc[i] <= 0.1 and 
			df['总氮(mg/L)'].loc[i] <= 0.5):
			standard = 2
			standard_list.append(standard)
			continue
		elif (df['氨氮(mg/L)'].loc[i] <= 1.0 and 
			df['溶解氧(mg/L)'].loc[i] >= 5.0 and 
			df['CODmn(mg/L)'].loc[i] <= 6.0 and 
			df['总磷(mg/L)'].loc[i] <= 0.2 and 
			df['总氮(mg/L)'].loc[i] <= 1.0):
			standard = 3
			standard_list.append(standard)
			continue
		elif (df['氨氮(mg/L)'].loc[i] <= 1.5 and 
			df['溶解氧(mg/L)'].loc[i] >= 3.0 and 
			df['CODmn(mg/L)'].loc[i] <= 10.0 and 
			df['总磷(mg/L)'].loc[i] <= 0.3 and 
			df['总氮(mg/L)'].loc[i] <= 1.5):
			standard = 4
			standard_list.append(standard)
			continue
		else:
			standard = 5
			standard_list.append(standard)
	#for 


	df['standard'] = standard_list
	return df



def train_test_data(df,train_start_date='2012-05-01',train_end_date='2016-04-30',test_strat_date='2016-05-01',test_end_date='2017-04-30'):
	start_date = datetime.datetime.strptime(start_date,'%Y-%m-%d')
	end_date = datetime.datetime.strptime(end_date,'%Y-%m-%d')
	mask_train = (df['记录时间'] > train_start_date) & (df['记录时间'] <= train_end_date)
	mask_test = (df['记录时间'] > test_strat_date) & (df['记录时间'] <= test_end_date)
	print ('训练数据集时间为{}至{}'.format(train_start_date,train_end_date))
	df_train = df.loc[mask_train]
	print ('训练数据集时间为{}至{}'.format(test_strat_date,test_end_date))
	df_test = df.loc[mask_test]
	return df_train,df_test


if __name__ == '__main__':
	import seaborn as sns
	#sns.set_style("darkgrid",{"font.sans-serif":['simhei', 'Arial']})
	import matplotlib.pyplot as plt 
	from matplotlib.font_manager import FontProperties
	font = FontProperties(fname=r"simhei.ttf")



	excel_path = 'your_excel'
	WQ_name = ['总氮','总磷','COD','氨氮','叶绿素']
	df,WQId_name = screen_excel(excel_path,WQ_name)
	#df2 = df[WQId_name]
	'''
	ax = df2.plot(figsize=(16,12))
	labels = ax.get_xticklabels()+ax.legend().texts+[ax.title]
	for label in labels : 
		label.set_fontproperties(font)
	plt.show()
	

	for name in WQId_name:
		df_name = df[[name]]
		df_plot(df_name)

	df3 = df3.drop(df3[[0]].idxmax())
	df3[[0]].max()

	'''

	df_train,df_test = train_test_data(df)