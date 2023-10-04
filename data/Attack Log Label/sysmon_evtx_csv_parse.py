from collections import Counter   
import json
import datetime
import Evtx.Evtx as evtx
import Evtx.Views as e_views
import networkx as nx
import pandas as pd
import numpy as np
import time

file_name='/Users/zhanghangsheng/Documents/my_code/logpedia/data/Attack Log Label/WINDOWS Hacker Attack Log/5/计划任务'
evtx_file = file_name+".evtx"
json_file = file_name+".json"
csv_file=file_name+".csv"

def parse_record(record=None, epochconvert=True):
    if record is None:
        return False
    temp = {}
    for node in record:
        parent = node.tag.split("}")[-1]
        for child in node:
            if parent == "EventData":
                if "Name" in child.attrib:
                    event_data_type = child.attrib["Name"]
                    temp["{}_{}".format(parent, event_data_type)] = child.text
                else:
                    if not "{}_rawdata".format(parent) in temp:
                        temp["{}_rawdata".format(parent)] = child.text
                    elif child.text is not None:
                        temp["{}_rawdata".format(parent)] = temp ["{}_rawdata".format(parent)]  + child.text
            else:
                child_name = child.tag.split("}")[-1]
                if child.attrib:
                    for key, value in child.attrib.items():
                        temp["{}_{}".format(child_name, key)] = value
                    temp[child_name] = child.text
                else:
                    temp[child_name] = child.text
    # time to epoch
    timekeys = ["EventData_UtcTime", "TimeCreated_SystemTime"]
    if epochconvert:
        for tkey in timekeys:
            if tkey in temp:
                try:
                    time = datetime.datetime.strptime(temp[tkey], "%Y-%m-%d %H:%M:%S.%f")
                except:
                    time = datetime.datetime.strptime(temp[tkey], "%Y-%m-%d %H:%M:%S")
                # temp[tkey] = int(time.strftime("%s"))
    return temp

j=open(json_file, "w") 
with evtx.Evtx(evtx_file) as log:
    for record in log.records():
        parsed_rec = parse_record(record=record.lxml(), epochconvert=True)
        json.dump(parsed_rec,j)
        j.write('\n')
j.close()
        
def count_arr(arr):
    counter = Counter(arr)
    # iterate over the counter and print the number and its count
    sorted_counts = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    # iterate over the counter and print the number and its count
    for num, count in sorted_counts:
        print(f"{num}: {count}")
        
G = nx.MultiDiGraph(name='zhs', data=True, align='vertical')


json_data=[]
for line in open(json_file, 'r', encoding='ISO-8859-1'):
    json_data.append(line)
EventID_arr=[]
# Event ID 1:进程创建Process Creation
Event1_arr=[]
# Event ID 3:网络连接Network Connections
Event3_arr=[]
# Event ID 7:image加载Image Loaded
Event7_arr=[]
# EventID 10: ProcessAccess
Event10_arr=[]
# Event ID 11:文件创建事件File Creation Events
Event11_arr=[]
# RegObject添加/删除(HKLM / HKU)
Event12_arr=[]
# Registry value set
# RegValue设置(DWORD / QWORD添加)
Event13_arr=[]
# RegObject重命名
Event14_arr=[]
# 事件ID 17:已创建管道
Event17_arr=[]
# 事件ID 18:管道连接
Event18_arr=[]
# Event ID 22: DNSEvents
Event22_arr=[]
# Event ID 23:FileDelete
Event23_arr=[]

Event_df=pd.DataFrame(np.empty((0, 8)),columns=['Time','EventID','ParentProcessGuid','ParentProcessID','ProcessGuid','ProcessID','ParentImage','Image'])

df = pd.DataFrame(columns=['source_pid','des_pid','time', 'relationship','source', 'destination'])

for i in range(len(json_data)):
    data = json.loads(json_data[i])
    if data['EventID']=='1':
        Event1_arr.append("EventID:1 "+str(data['EventData_UtcTime'])+" "+data['EventData_ParentImage']+"------"+data['EventData_Image'])
        new_row = pd.Series({'Time':str((data['EventData_UtcTime'])),'EventID':data['EventID'],'ParentProcessGuid':data['EventData_ParentProcessGuid'],'ParentProcessID':data['EventData_ParentProcessId'],'ProcessGuid':data['EventData_ProcessGuid'],'ProcessID':data['EventData_ProcessId'],'ParentImage':data['EventData_ParentImage'],'Image':data['EventData_Image']})
        Event_df= Event_df.append(new_row, ignore_index=True)
        row = {'source': data['EventData_ParentImage'],'source_pid':data['EventData_ParentProcessId'],'des_pid':data['EventData_ProcessId'], 'destination': data['EventData_Image'], 'relationship': 'process_create', 'time': data['EventData_UtcTime']}
        df.loc[len(df)] = row
        
    if data['EventID']=='3':
        Event3_arr.append("EventID:3 "+str(data['EventData_UtcTime'])+" "+str(data['EventData_SourcePort'])+"--"+data["EventData_Protocol"]+"--"+data['EventData_DestinationIp']+'--'+str(data['EventData_DestinationPort']))
    if data['EventID']=='7':
        Event7_arr.append("EventID:7 "+str(data['EventData_UtcTime'])+" "+data['EventData_Image']+"------"+data['EventData_ImageLoaded'])
        new_row = pd.Series({'Time':str((data['EventData_UtcTime'])),'EventID':data['EventID'],'ParentProcessGuid':data['EventData_ProcessGuid'],'ParentProcessID':data['EventData_ProcessId'],'ProcessGuid':'','ProcessID':'','ParentImage':data['EventData_Image'],'Image':data['EventData_ImageLoaded']})
        Event_df= Event_df.append(new_row, ignore_index=True)   
        row = {'source': data['EventData_Image'],'source_pid':data['EventData_ProcessId'],'des_pid':data['EventData_ProcessId'], 'destination': data['EventData_ImageLoaded'], 'relationship': 'load', 'time': data['EventData_UtcTime']}
        df.loc[len(df)] = row
    if data['EventID']=='10':
        Event10_arr.append("EventID:10 "+str(data['EventData_UtcTime'])+" "+data['EventData_SourceImage']+"-----"+data['EventData_TargetImage'])
        new_row = pd.Series({'Time':str((data['EventData_UtcTime'])),'EventID':data['EventID'],'ParentProcessGuid':data['EventData_SourceProcessGUID'],'ParentProcessID':data['EventData_SourceProcessId'],'ProcessGuid':data['EventData_TargetProcessGUID'],'ProcessID':data['EventData_TargetProcessId'],'ParentImage':data['EventData_SourceImage'],'Image':data['EventData_TargetImage']})
        Event_df= Event_df.append(new_row, ignore_index=True)
        row = {'source': data['EventData_SourceImage'],'source_pid':data['EventData_SourceProcessId'],'des_pid':data['EventData_TargetProcessId'] ,'destination': data['EventData_TargetImage'], 'relationship': 'access', 'time': data['EventData_UtcTime']}
        df.loc[len(df)] = row
    if data['EventID']=='11':
        Event11_arr.append("EventID:11 "+str(data['EventData_UtcTime'])+" "+data['EventData_Image']+"-----"+data['EventData_TargetFilename'])
        row = {'source': data['EventData_Image'],'source_pid':data['EventData_ProcessId'],'des_pid':data['EventData_ProcessId'], 'destination': data['EventData_TargetFilename'], 'relationship': 'file_create', 'time': data['EventData_UtcTime']}
        df.loc[len(df)] = row
    if data['EventID']=='12':
        Event12_arr.append("EventID:12 "+str(data['EventData_UtcTime'])+" "+data['EventData_Image']+"-----"+data['EventData_TargetObject'])
        row = {'source': data['EventData_Image'],'source_pid':data['EventData_ProcessId'],'des_pid':data['EventData_ProcessId'], 'destination': data['EventData_TargetObject'], 'relationship': 'reg_create_del', 'time': data['EventData_UtcTime']}
        df.loc[len(df)] = row
    if data['EventID']=='13':
        Event13_arr.append("EventID:13 "+str(data['EventData_UtcTime'])+" "+data['EventData_Image']+"-----"+data['EventData_TargetObject'])
        row = {'source': data['EventData_Image'], 'source_pid':data['EventData_ProcessId'],'des_pid':data['EventData_ProcessId'],'destination': data['EventData_TargetObject'], 'relationship': 'reg_set_value', 'time': data['EventData_UtcTime']}
        df.loc[len(df)] = row
    if data['EventID']=='14':
        Event14_arr.append("EventID:14 "+str(data['EventData_UtcTime'])+" "+data['EventData_Image']+"-----"+data['EventData_TargetObject'])
        row = {'source': data['EventData_Image'],'source_pid':data['EventData_ProcessId'],'des_pid':data['EventData_ProcessId'], 'destination': data['EventData_TargetObject'], 'relationship': 'reg_rename', 'time': data['EventData_UtcTime']}
        df.loc[len(df)] = row
    if data['EventID']=='17':
        Event17_arr.append("EventID:17 "+str(data['EventData_UtcTime'])+" "+data['EventData_Image']+"-----"+data['EventData_PipeName'])
    if data['EventID']=='18':
        Event18_arr.append("EventID:18 "+str(data['EventData_UtcTime'])+" "+data['EventData_Image']+"-----"+data['EventData_PipeName'])
    if data['EventID']=='22':
        Event22_arr.append("EventID:22 "+str(data['EventData_UtcTime'])+" "+data['EventData_Image']+"-----"+data['EventData_QueryName'])
    if data['EventID']=='23':
        Event23_arr.append("EventID:23 "+str(data['EventData_UtcTime'])+" "+data['EventData_Image']+"-----"+data['EventData_TargetFilename'])
        row = {'source': data['EventData_Image'],'source_pid':data['EventData_ProcessId'],'des_pid':data['EventData_ProcessId'], 'destination': data['EventData_TargetFilename'], 'relationship': 'file_del', 'time': data['EventData_UtcTime']}
        df.loc[len(df)] = row
        
    if "EventID" in data:
        EventID_arr.append(data['EventID'])
   
# print(df)
df.to_csv(csv_file)




