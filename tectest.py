import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import locale
import csv
import seaborn as sns
sns.set(style='ticks')
# digest table
all_event = pd.read_csv(
    '/Users/apple/Documents/traffic_engineering/Events_Crashes.csv', encoding='latin-1')
# select driver who reacte the time
react_event = all_event[all_event['Reaction Start'] > 0]
non_react_event = all_event[all_event['Reaction Start'] < 0]
cellphone_texting = all_event[all_event['Sec Task 1'] == 'Cell phone, Texting']

cellphone_all = all_event[(
    all_event['Sec Task 1'] == 'Cell phone, Texting')
    | (all_event['Sec Task 1'] == 'Cell phone, Browsing')
    | (all_event['Sec Task 1'] == 'Cell phone, Locating/reaching/answering')
    | (all_event['Sec Task 1'] == 'Cell phone, other')]

cellphone_all1 = react_event[(
    react_event['Sec Task 1'] == 'Cell phone, Texting')
    | (react_event['Sec Task 1'] == 'Cell phone, Browsing')
    | (react_event['Sec Task 1'] == 'Cell phone, Locating/reaching/answering')]

# !!!!!!!switch the data set we want to research:event!!!!!!!!!!!

event = cellphone_all1


print(event.head(5))

'''
print(event.head(4))
print(type(event))
print(event[['Reaction Start', 'Impact Time']].describe())
print(type(event[['Reaction Start']]))
print(event[['Reaction Start']].describe())
'''
# sucstract reation time
rt = event['Reaction Start'] - event['Event Start']
print(rt)
# millescond to scond , unit change
event['rt'] = rt/1000

print(event[['rt']].describe())
# sucstract impact time - reaction Time
irt = event['Impact Time'] - event['Reaction Start']
print(irt)
event['irt'] = irt/1000

print(event[['irt']].describe())


'''
# plt.plot(event[['Reaction Start']], event[['Impact Time']])

'''
# crash severity vs rt and irt(the relationship beween irt and rt)
'''
fg = sns.FacetGrid(data=event, hue='Crash Severity 1', aspect=1.61)
fg.map(plt.scatter, 'rt', 'irt').add_legend()

'''
# cat plot violin to draw driving behavior , rt and event severity (two lelvel:crash,near_crash)
'''
fig, ax = plt.subplots()
ax.chart = sns.catplot(x='Driving Behavior 1', y='irt', hue='Event Severity 1',
                       kind='violin', split=True, data=event)
'''

'''
plt.draw()
plt.xticks(rotation=30)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')

'''
# plt.scatter(event['rt'], event['irt'], c=(0, 0, 0), alpha=0.5)

'''

sns.boxplot(x="Crash Severity 1", y="rt", hue="Driving Behavior 1", data=event,
            linewidth=0.5)
'''

'''
# plot cellphone_all_related value about rt & irt
sns.boxplot(x="Sec Task 1", y="irt", hue="Crash Severity 1", data=event,
            linewidth=0.5)
        '''
# cellphone_all Surface Cndtn_about rt & irt
'''
sns.boxplot(x="Surface Cndtn", y="rt", hue="Sec Task 1", data=event,
            linewidth=0.5)
        '''
# cellphone_all Light  about rt & irt
'''
sns.boxplot(x="Light", y="irt", hue="Sec Task 1", data=event,
            linewidth=0.5)
'''
# cell_phone  Traffic Flow (seperation method)    rt& irt
'''
sns.boxplot(x="Traffic Flow", y="rt", hue="Sec Task 1", data=event,
            linewidth=0.5)
'''
# cell phone Traffic Control  rt& irt
'''
sns.boxplot(x="Traffic Control", y="irt", hue="Sec Task 1", data=event,
            linewidth=0.5)
'''
'''
# cell phone Grade rt& irt
sns.boxplot(x="Grade", y="rt", hue="Sec Task 1", data=event,
            linewidth=0.5)
'''
# cell phone Locality rt& irt
'''
sns.boxplot(x="Locality", y="irt", hue="Sec Task 1", data=event,
            linewidth=0.5)
plt.xticks(rotation=30)
'''
# sns.distplot(rt)
# how the only mean value and its confidence interval within each nested category:
'''
sns.catplot(x='Driving Behavior 1', y='rt', hue='Event Severity 1',
            kind='bar', data=event)


plt.xticks(rotation=30)

N = 12
ind = np.arange(N)
width = 0.35

p1 = plt.bar(ind,)
'''
# how the only mean value and its confidence interval within each nested category:
'''
sns.catplot(x='Sec Task 1', y='irt', hue='Event Severity 1',
            kind='bar', data=event)
plt.xticks(rotation=30)
'''

plt.show()
