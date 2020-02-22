import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import locale
import csv
# digest table
event = pd.read_csv(
    '/Users/apple/Documents/traffic_engineering/event_five_record_sample1.csv', encoding='latin-1')
# show evets's id

'''Events.Event_ID
# show all column's name
event.columns.values
# show sec task1's value;bacause it's important to see the distracted bahaviour
event['Sec Task 1']
# show first row's value
event.loc[0]
# show the whole length of columns
event.index


event.set_index('Sec Task 1', inplace=True)
# how many rows and columns and other information
Events.info()
# how many rows
len(event)

'''


# sec tack1 info from record 1 - 30
# print(event.iloc[0:30]['Sec Task 1'])

# print(event.head())
# try something new
# pp.plot(event.iloc[1:5]['Event Start']

# substract two rows
# print(event.dtypes)


# convert str with commas in it as thousands seperators

# create a row "new" denote the time between reaction start and event start
'''
for i in range(0, event_length):
    event = event.assign(ett =locale.atoi(event.iloc[i]['Event Start']))
    b = locale.atoi(event.iloc[i]['Reaction Start'])



    c = b - a
    def fn(row): return c
    col = event.apply(fn, axis=1)
    event = event.assign(new=col.values)
    print(event)'''

'''event_length = len(event.index)
print(event_length)'''
# create the list to store the clean data
# Est:event start
# Rst: reaction Time
# Itt:impact time
# Rt:reaction time duration(reaction Time - event start)
# Tast1:Sec Task 1 Start Time
# Taet1:Sec Task 1 End Time
# Tast2:Sec Task 2 Start Time
# Taet2:Sec Task 2 End Time
# Tast3:Sec Task 3 Start Time
# Taet3:Sec Task 3 End Time

'''Est = []
Rst = []
Itt = []
Rt = []
Tast1 = []
Taet1 = []
Tast2 = []
Taet2 = []
Tast3 = []
Taet3 = []'''
#test = []
'''locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')'''
'''
for i in range(0, event_length):
    Est.append(event.apply(lambda x: locale.atoi(event.iloc[i]['Event Start'])))
    #Est.append(event['Est'].apply(lambda x: locale.atoi(event.iloc[i]['Event Start'])))
    #print(locale.atoi(event.iloc[i]['Event Start']))
    Rst.append(event.apply(lambda x: locale.atoi(
        event.iloc[i]['Reaction Start'])))
    Itt.append(event.apply(lambda x: locale.atoi(event.iloc[i]['Impact Time'])))
    Tast1.append(event.apply(lambda x: locale.atoi(event.iloc[i]['Sec Task 1 Start Time'])))
    Taet1.append(event.apply(lambda x: locale.atoi(event.iloc[i]['Sec Task 1 End Time'])))
    Tast2.append(event.apply(lambda x: locale.atoi(event.iloc[i]['Sec Task 2 Start Time'])))
    Taet2.append(event.apply(lambda x: locale.atoi(event.iloc[i]['Sec Task 2 End Time'])))
    Tast3.append(event.apply(lambda x: event.iloc[i]['Sec Task 3 Start Time']))
    Taet3.append(event.apply(lambda x: event.iloc[i]['Sec Task 3 End Time']))
    Rt.append(event.apply(lambda x: locale.atoi(
        event.iloc[i]['Reaction Start']) - locale.atoi(event.iloc[i]['Event Start'])))
'''
'''# create a column from the list
event['Est'] = Est
event['Rst'] = Rst
event['Itt'] = Itt
event['Rt'] = Rt
event['Tast1'] = Tast1
event['Tast2'] = Tast2
event['Tast3'] = Tast3
event['Taet1'] = Taet1
event['Taet2'] = Taet2
event['Taet3'] = Taet3
#event['test'] = test
'''
'''
event[['Event Start', 'Reaction Start', 'Impact Time']] = event[[
    'Event Start', 'Reaction Start', 'Impact Time']].apply(pd.to_numeric, errors='ignore')
print(event[['Event Start']].describe())

print(event['Event Start'])

print(event['Event Start'].dtype)
print(type(event['Event Start']))
'''
# print(event[['Rt', 'Rst', 'Itt', 'Rt', 'Tast1', 'Est']])
'''
# try to write to a csv file with new columns
with open('myinenenen.csv', 'w', newline='') as file:
    writer =
'''
'''
# try to see the table with 5 rows for runnning fast
print(event[['Rt']].describe())
print(event.columns)
print(event)

'''
'''
print(event['Est'])
print(type(event['Est']))
event['Est'].astype(int)
'''
'''
event.plot(
    x='Est', y='Rt', color='red'
)
plt.show
'''

'''
print(event['Est'])
# print(event['test'])
print(event['Event Start'])
'''
