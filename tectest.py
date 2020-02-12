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
    '/Users/apple/Documents/traffic_engineering/Events_big_set_after_name.csv', encoding='latin-1')
# select driver who reacte the time
react_event = all_event[all_event['Reaction Start'] > 0]
non_react_event = all_event[all_event['Reaction Start'] < 0]

# switch the data set we want to research:event
event = react_event

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
# sns.distplot(rt)
# how the only mean value and its confidence interval within each nested category:
sns.catplot(x='Driving Behavior 1', y='rt', hue='Event Severity 1',
            kind='bar', data=event)

plt.xticks(rotation=30)

N = 12
ind = np.arange(N)
width = 0.35

p1 = plt.bar(ind,)
plt.show()
