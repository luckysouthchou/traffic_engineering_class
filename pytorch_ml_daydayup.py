
# ======begin with data exploration======
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# digest table
'''
# small_sample_data_route(for test and debug)
all_event = pd.read_csv(
    '/Users/apple/Documents/traffic_engineering/event_five_record_sample.csv', encoding='latin-1')

'''

# big_data_sample_route(for test and debug)
all_event = pd.read_csv(
    '/Users/apple/Documents/traffic_engineering/Events_Crashes.csv', encoding='latin-1')


'''
# original_data_route
all_event = pd.read_csv(
    '/Users/apple/Documents/traffic_engineering/data_set/Events888.csv', encoding='latin-1')
'''

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


# ====the begin of deep learning learning====

# use event standing for the data we interested

event = event_related_event_subject


# sucstract reation time
# test a small piece of event time set if they right value type
#rt = event.iloc[430:450]['Reaction Start'] - event.iloc[430:450]['Event Start']

rt = event.iloc[:]['Reaction Start'] - event.iloc[:]['Event Start']

rt = event['Reaction Start'] - event['Event Start']

# =======问题===
#print(event.iloc[430:450]['Participant_ID', 'Reaction Start'])


print(rt)
# millescond to scond , unit change
event['rt'] = rt/1000

print(event[['rt']].describe())
# sucstract impact time - reaction Time
irt = event['Impact Time'] - event['Reaction Start']
print(irt)
event['irt'] = irt/1000

print(event[['irt']].describe())


# 'Event_ID', 'Participant_ID' ang other columns related to description events will not be impact the result, so we will not use
# choose categorical values

# categorical_columns = ['Pre-Incident Maneuver', 'Precipitating Event', 'Event Nature 1', 'Impairments', 'Sec Task 1', 'Hands on Wheel', 'Driver Seatbelt', 'Infrastructure', 'Vis Obstructions',
#                       'Light', 'Weather', 'Surface Cndtn', 'Traffic Flow', 'Traffic Density', 'Traffic Control', 'Relation to Junction', 'Intersection Influence', 'Rd Alignment', 'Grade', 'Locality', 'Construction Zone']

categorical_columns = ['Sec Task 1', 'Light', 'Weather', 'Locality']

numerical_columns = ['rt', 'irt']

outputs = ['Event Severity 1']
for i in outputs:
    event[i] = event[i].astype('category')


print(event.dtypes)
# convert categorical value to category
for category in categorical_columns:
    event[category] = event[category].astype('category')

print(event['Light'].cat.categories)
print(event['Light'].head())
print(event['Light'].head().cat.codes)
print(event['rt'].head())


# convert categorical columns to tensors
task = event['Sec Task 1'].cat.codes.values
light = event['Light'].cat.codes.values
weather = event['Weather'].cat.codes.values
locality = event['Locality'].cat.codes.values


categorical_data = np.stack([task, light, weather, locality], 1)

print(categorical_data)

# convert  categorical value to tensors
categorical_data = torch.tensor(categorical_data, dtype=torch.int64)
print(categorical_data)

# convert numerical value to tensors
numerical_data = np.stack([event[col].values for col in numerical_columns], 1)
# convert numerical value to tensors
numerical_data = torch.tensor(numerical_data, dtype=torch.float)
print(numerical_data)

# convert output to tensors

outputs = event['Event Severity 1'].cat.codes.values
#outputs = np.stack(['Event Severity 1'], 1)
outputs = torch.tensor(outputs, dtype=torch.int64)
print(outputs)


# print the shape of categorical data, numerical data, and their out inputs
print(categorical_data.shape)
print(numerical_data.shape)
print(outputs.shape)


# create tuple for categorical value(transfer to vector)
categorical_columns_size = [len(event[colum].cat.categories) for colum in categorical_columns]
categorical_embedding_size = [(col_size, min(50, (col_size+1)//2))
                              for col_size in categorical_columns_size]
print(categorical_embedding_size)


# dividin our dataset to training set and testings set0.8 , 0.2
total_records = 1005

test_records = int(total_records * .2)
categorical_train_data = categorical_data[:total_records-test_records]
categorical_test_data = categorical_data[total_records-test_records:total_records]
numerical_train_data = numerical_data[:total_records-test_records]
numerical_test_data = numerical_data[total_records-test_records:total_records]
train_outputs = outputs[:total_records-test_records]
test_outputs = outputs[total_records-test_records:total_records]

# verify dataset size
print(len(categorical_train_data))
print(len(numerical_train_data))
print(len(train_outputs))

print(len(categorical_test_data))
print(len(numerical_test_data))
print(len(test_outputs))


# model for prediction
class Model(nn.Module):

    def __init__(self, embedding_size, num_numerical_cols, output_size, layers, p=0.4):
        super().__init__()
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])
        self.embedding_dropout = nn.Dropout(p)
        self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)

        all_layers = []
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols + num_numerical_cols

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_categorical, x_numerical):
        embeddings = []
        for i, e in enumerate(self.all_embeddings):
            embeddings.append(e(x_categorical[:, i]))
        x = torch.cat(embeddings, 1)
        x = self.embedding_dropout(x)

        x_numerical = self.batch_norm_num(x_numerical)
        x = torch.cat([x, x_numerical], 1)
        x = self.layers(x)
        return x

# training the model


model = Model(categorical_embedding_size, numerical_data.shape[1], 2, [200, 100, 50], p=0.4)


print(model)


# define loss function, because it's classification problem, use cross entropy loss
loss_function = nn.CrossEntropyLoss()

# define optimizer, choose adam , learning rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# begin to train the model , set epoch
epochs = 300
aggregated_losses = []

for i in range(epochs):
    i += 1
    y_pred = model(categorical_train_data, numerical_train_data)
    single_loss = loss_function(y_pred, train_outputs)
    aggregated_losses.append(single_loss)

    if i % 25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    optimizer.zero_grad()
    single_loss.backward()
    optimizer.step()

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')


# plot the losses against epochs

plt.plot(range(epochs), aggregated_losses)
plt.xlabel('Loss')
plt.ylabel('epoch')
# plt.show()


# making prediction using test data , pass test categorical data  and test numerical data to model
# print rhe prediction value for test data and print cross entropy loss for test data set0

with torch.no_grad():
    y_val = model(categorical_test_data, numerical_test_data)
    loss = loss_function(y_val, test_outputs)
print(f'Loss: {loss:.8f}')

# because the output layer contains 2 neurons, each prediction  contains 2 value,
# so, print outputs
print(y_val[:5])
y_val = np.argmax(y_val, axis=1)


# use confusion_matrix, accuracy_score, classification_report to see the accuracy...


print(confusion_matrix(test_outputs, y_val))
print(classification_report(test_outputs, y_val))
print(accuracy_score(test_outputs, y_val))
