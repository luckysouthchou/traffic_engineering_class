
# ======begin with data exploration======
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# digest table
'''
# small_sample_data_route(for test and debug)
all_event = pd.read_csv(
    '/Users/apple/Documents/traffic_engineering/event_five_record_sample.csv', encoding='latin-1')
'''

'''
# big_data_sample_route(for test and debug)
all_event = pd.read_csv(
    '/Users/apple/Documents/traffic_engineering/Events_Crashes.csv', encoding='latin-1')
'''


# original_data_route
all_event = pd.read_csv(
    '/Users/apple/Documents/traffic_engineering/data_set/Events.csv', encoding='latin-1')


# select the crash_relevent_data_subject nor non_subject

event_related_event_subject = all_event[(
    all_event['Event Severity 1'] == 'Crash')
    | (all_event['Event Severity 1'] == 'Near-Crash')
    | (all_event['Event Severity 1'] == 'Baseline')
    | (all_event['Event Severity 1'] == 'Crash-Relevant')]

# set crash related events
event_related_event_subject.loc[((event_related_event_subject['Event Severity 1'] == 'Crash')
                                 | (event_related_event_subject['Event Severity 1'] == 'Near-Crash')
                                 | (event_related_event_subject['Event Severity 1'] == 'Crash-Relevant')), 'Event Severity 1'] = 'crash_related'
# set non crash related Events values
event_related_event_subject.loc[event_related_event_subject['Event Severity 1']
                                == 'Baseline'] = 'non_crash'

# other : event_related_event_subject_non_using_cellphone
# change all subject other than cellphone_related to name:other
event_related_event_subject.loc[((event_related_event_subject['Sec Task 1'] != 'Cell phone, Texting')
                                 & (event_related_event_subject['Sec Task 1'] != 'Cell phone, Browsing')
                                 & (event_related_event_subject['Sec Task 1'] != 'Cell phone, Locating/reaching/answering')
                                 & (event_related_event_subject['Sec Task 1'] != 'Cell phone, other')
                                 & (event_related_event_subject['Sec Task 1'] != 'No Secondary Tasks')
                                 & (event_related_event_subject['Sec Task 1'] != 'Passenger in adjacent seat - interaction')
                                 & (event_related_event_subject['Sec Task 1'] != 'Other external distraction')
                                 & (event_related_event_subject['Sec Task 1'] != 'Talking/singing, not with passenger')
                                 & (event_related_event_subject['Sec Task 1'] != 'Other non-specific internal eye glance')), 'Sec Task 1'] = 'other distraction'


# set all cell phone related subject to have a new name cellphone related distraction
event_related_event_subject.loc[((event_related_event_subject['Sec Task 1'] == 'Cell phone, Texting')
                                 | (event_related_event_subject['Sec Task 1'] == 'Cell phone, Browsing')
                                 | (event_related_event_subject['Sec Task 1'] == 'Cell phone, Locating/reaching/answering')
                                 | (event_related_event_subject['Sec Task 1'] == 'Cell phone, other')), 'Sec Task 1'] = 'cellphone related distraction'


# selct task related to cellphone
cellphone_all_event_subject = event_related_event_subject[(
    event_related_event_subject['Sec Task 1'] == 'Cell phone, Texting')
    | (event_related_event_subject['Sec Task 1'] == 'Cell phone, Browsing')
    | (event_related_event_subject['Sec Task 1'] == 'Cell phone, Locating/reaching/answering')
    | (event_related_event_subject['Sec Task 1'] == 'Cell phone, other')]


'''
# print pie chart related to all distracted Events
event = cellphone_all_event_subject


print(event_related_event_subject['Event Severity 1'])

# print crash vs non_crash bar plot
sns.countplot(x='Event Severity 1', data=event_related_event_subject)

plt.xticks(rotation=30)

plt.show()

# check if the value of 'seck task 1 changed'
print(event_related_event_subject['Sec Task 1'])

# bar plot other distracted behaviour vs cell phone bahaviour
event_related_event_subject['Sec Task 1'].value_counts().plot(kind='pie', autopct='%1.1f%%')

plt.show()


event['Sec Task 1'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=[
    'skyblue', 'orange', 'red', 'green', 'black'], explode=(0.06, 0.03, 0.01, 0.01, 0.01))


# shape the datadet
print(event.shape)

# head five rows
print(event.head())


# explore data analysis-------

# create the default size for graphs
fig_size = plt.rcParams['figure.figsize']
fig_size[0] = 10
fig_size[1] = 8
plt.rcParams['figure.figsize'] = fig_size


#  pie plot for the Exited column
event['Event Severity 1'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=[
    'skyblue', 'orange', 'red', 'green'], explode=(0.06, 0.03, 0.01, 0.02))


# bar plot phone using situation with all events including baseline and accident
sns.countplot(x='Sec Task 1', data=event)
plt.xticks(rotation=30)

# phone using situation with all events including accident relevate
sns.countplot(x='Sec Task 1', data=event[event['Event Severity 1'] == 'Crash'])
event['Event Severity 1'] != 'Baseline'
plt.xticks(rotation=30)

# plot bar chart to see the relationship between cell phone and other behaviour with crash
sns.countplot(x='Event Severity 1', hue='Sec Task 1', data=event_related_event_subject)
plt.xticks(rotation=30)


plt.show()
'''

# =======the end of data exploration==========

# ====the begin of machine learning====

# use event standing for the data we interested
event = event_related_event_subject

print(event.columns)
# 'Event_ID', 'Participant_ID' ang other columns related to description events will not be impact the result, so we will not use
# choose categorical values

categorical_columns = ['Pre-Incident Maneuver', 'Precipitating Event', 'Event Nature 1', 'Impairments', 'Sec Task 1', 'Hands on Wheel', 'Driver Seatbelt', 'Infrastructure', 'Vis Obstructions',
                       'Light', 'Weather', 'Surface Cndtn', 'Traffic Flow', 'Traffic Density', 'Traffic Control', 'Relation to Junction', 'Intersection Influence', 'Rd Alignment', 'Grade', 'Locality', 'Construction Zone']


# sucstract reation time
# test a small piece of event time set if they right value type
#rt = event.iloc[430:450]['Reaction Start'] - event.iloc[430:450]['Event Start']
#rt = event['Reaction Start'] - event['Event Start']

# =======问题===
print(event.iloc[430:450]['Participant_ID', 'Reaction Start'])


'''
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
