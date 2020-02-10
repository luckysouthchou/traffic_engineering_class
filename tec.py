import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as pp
# digest table
event = pd.read_csv('/Users/apple/Documents/traffic_engineering/Events.csv', encoding='latin-1')
# show evets's id

'''Events.Event_ID
# show all column's name
Events.columns.values
# show sec task1's value;bacause it's important to see the distracted bahaviour
Events['Sec Task 1']
# show first row's value
Events.loc[0]
# show the whole length of columns
Events.index

#test
#pppp
Events.set_index('Sec Task 1', inplace=True)
# how many rows and columns and other information
Events.info()
# how many rows
len(Events)

'''


# sec tack1 info from record 1 - 30
# print(event.iloc[0:30]['Sec Task 1'])

# print(event.head())
# try something new
# pp.plot(event.iloc[1:5]['Event Start']

# substract two rows
print(event.dtypes)
#test11 = event.iloc[0]['Impact Time'].astype(float) - event.iloc[0]['Event Start'].astype(float)
