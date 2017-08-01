from __future__ import print_function
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
import csv
import math
import scipy.stats as st
import matplotlib.pyplot as plt
import kmedoids
from datetime import datetime
import numpy as np
from scipy import stats
from math import pi
import random
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm


def read_attributes(filename):
    ids = []
    attributes = []
    attributes_complete = []
    with open(filename, 'Ur') as csvfile:
        reader = csv.reader(csvfile)
        l_n = 0
        for row in reader:
            if (l_n == 0):
                list_label = row
            if (l_n != 0):
                # deal with categorical data
                for i in range(0, 2):
                    row.insert(5, 0)
                for i in range(0, 4):
                    row.insert(8, 0)
                for i in range(0, 2):
                    row.insert(13, 0)
                for i in range(len(row)):
                    if (row[i] == 'Transfer'):
                        row[i] = 1
                    elif (row[i] == 'Stat'):
                        row[i] = 0
                        row[i + 1] = 1
                    elif (row[i] == 'Attending'):
                        row[i] = 0
                        row[i + 2] = 1

                    if (row[i] == 'Blunt'):
                        row[i] = 1
                    if (row[i] == 'Penetrating'):
                        row[i] = 0
                        row[i + 1] = 1
                    if (row[i] == 'Animal Bite'):
                        row[i] = 0
                        row[i + 2] = 1
                    if (row[i] == 'Burn'):
                        row[i] = 0
                        row[i + 3] = 1
                    if (row[i] == 'Blunt/No Injury'):
                        row[i] = 0
                        row[i + 4] = 1

                    if (row[i] == 'Non-critical admission'):
                        row[i] = 1
                    if (row[i] == 'Critical Admission'):
                        row[i] = 0
                        row[i + 1] = 1
                    if (row[i] == 'Discharged'):
                        row[i] = 0
                        row[i + 2] = 1
                attributes.append(row)
                temp = row[:]
                attributes_complete.append(temp)
            l_n += 1
        for row in attributes:
            for i in range(len(row) - 1, -1, -1):
                if row[i] == 'Yes':
                    del row[i]
            ids.append(row[0])
            del row[0]
        list_label[4] = 'Transfer'
        list_label.insert(5, 'Attending')
        list_label.insert(5, 'Stat')
        list_label[7] = 'Blunt'
        list_label.insert(8, 'Blunt/No Injury')
        list_label.insert(8, 'Burn')
        list_label.insert(8, 'Animal Bite')
        list_label.insert(8, 'Penetrating')
        list_label[12] = 'Non-critical admission'
        list_label.insert(13, 'Discharged')
        list_label.insert(13, 'Critical Admission')
    return attributes, ids, list_label, attributes_complete


def read_attributes_weight(filename, weights):
    ids = []
    attributes = []
    attributes_complete = []
    with open(filename, 'Ur') as csvfile:
        reader = csv.reader(csvfile)
        l_n = 0
        for row in reader:
            if (l_n == 0):
                list_label = row
            if (l_n != 0):
                # deal with categorical data
                for i in range(0, 2):
                    row.insert(5, 0)
                for i in range(0, 4):
                    row.insert(8, 0)
                for i in range(0, 2):
                    row.insert(13, 0)
                for i in range(len(row)):
                    if (row[i] == 'Transfer'):
                        row[i] = 1
                    elif (row[i] == 'Stat'):
                        row[i] = 0
                        row[i + 1] = 1
                    elif (row[i] == 'Attending'):
                        row[i] = 0
                        row[i + 2] = 1

                    if (row[i] == 'Blunt'):
                        row[i] = 1
                    if (row[i] == 'Penetrating'):
                        row[i] = 0
                        row[i + 1] = 1
                    if (row[i] == 'Animal Bite'):
                        row[i] = 0
                        row[i + 2] = 1
                    if (row[i] == 'Burn'):
                        row[i] = 0
                        row[i + 3] = 1
                    if (row[i] == 'Blunt/No Injury'):
                        row[i] = 0
                        row[i + 4] = 1

                    if (row[i] == 'Non-critical admission'):
                        row[i] = 1
                    if (row[i] == 'Critical Admission'):
                        row[i] = 0
                        row[i + 1] = 1
                    if (row[i] == 'Discharged'):
                        row[i] = 0
                        row[i + 2] = 1
                attributes.append(row)
                temp = row[:]
                attributes_complete.append(temp)
            l_n += 1
        for row in attributes:
            for i in range(len(row) - 1, -1, -1):
                if row[i] == 'Yes':
                    del row[i]
            ids.append(row[0])
            del row[0]
        list_label[4] = 'Transfer'
        list_label.insert(5, 'Attending')
        list_label.insert(5, 'Stat')
        list_label[7] = 'Blunt'
        list_label.insert(8, 'Blunt/No Injury')
        list_label.insert(8, 'Burn')
        list_label.insert(8, 'Animal Bite')
        list_label.insert(8, 'Penetrating')
        list_label[12] = 'Non-critical admission'
        list_label.insert(13, 'Discharged')
        list_label.insert(13, 'Critical Admission')
        for row, row2 in zip(attributes, attributes_complete):
            for i in range(len(row)):
                row[i] = int(row[i]) * weights[i]
                row2[i+2] = int(row2[i+2]) * weights[i]
    return attributes, ids, list_label, attributes_complete


def kmeans(attributes, ids, n=2):

    # dimension reduction
    data = np.array(attributes)
    reduced_data = PCA(n_components=2).fit_transform(data)

    # predict
    kmeans = KMeans(n_clusters=n, random_state=0).fit(reduced_data)

    group_members = [[] for i in range(n)]
    num = 0
    for i in kmeans.labels_:
        group_members[i].append(ids[num])
        num+=1
    show_kmeans(kmeans, reduced_data)
    # return kmeans result and ids of patients for tracing
    return group_members, kmeans, reduced_data


def kmedoid(attributes, ids, n=2):
    # dimension reduction
    data = np.array(attributes)
    reduced_data = PCA(n_components=2).fit_transform(data)

    D = pairwise_distances(reduced_data, metric='euclidean')

    # split into 2 clusters
    # #M store the points that is regarded as center
    M, C = kmedoids.kMedoids(D, n)

    group_members = [[] for i in range(n)]
    for i in range(n):
        for j in C[i]:
            group_members[i].append(ids[j])
    show_kmedoid(M, C, reduced_data)
    # return kmeans result and ids of patients for tracing
    return group_members, M, C, reduced_data


def do_ngram(filename, group_members, duration_time=False):
    #define multi-dimension dictionaries
    act_times = {}
    act_times[0] = {}
    if len(group_members)>1:
        for i in range(len(group_members)):
            if i!=0:
                act_times[i] = {}

    acts = [[] for i in range(len(group_members))]

    if duration_time==False:
        result, act_times[0], acts[0] = ngram(filename, group_members[0], n=1)  # n is the length of sequence
        result.sort(reverse=True)
        # file to write result
        file = 'without_duration.csv'
        write_ngram(file, result, group_members[0], type="new")
        for i in range(1000):
            result, temp, temp2 = ngram(filename, group_members[0], i + 2)
            result.sort(reverse=True)
            if (len(result) == 0):
                break
            write_ngram(file, result, group_members[0], n=i + 2, type="append")
            act_times[0].update(temp)
            acts[0] = acts[0] + temp2
        for k in range(len(group_members)):
            if k != 0:
                result, act_times[k], acts[k] = ngram(filename, group_members[k], n=1)  # n is the length of sequence
                result.sort(reverse=True)
                write_ngram(file, result, group_members[k], type="new_group")
                for i in range(1000):
                    result, temp, temp2 = ngram(filename, group_members[k], i + 2)
                    result.sort(reverse=True)
                    if (len(result) == 0):
                        break
                    write_ngram(file, result, group_members[k], n=i + 2, type="append")
                    act_times[k].update(temp)
                    acts[k] = acts[k] + temp2
        return act_times, acts

    if duration_time==True:
        act_duration = {}
        act_duration[0] = {}
        if len(group_members) > 1:
            for i in range(len(group_members)):
                if i != 0:
                    act_duration[i] = {}
        result, act_times[0], acts[0], act_duration[0] = ngram(filename, group_members[0], n=1,
                                              duration_time=duration_time)  # n is the length of sequence
        for k in range(len(group_members)):
            if k != 0:
                result, act_times[k], acts[k], act_duration[k] = ngram(filename, group_members[k], n=1, duration_time=duration_time)  # n is the length of sequence
        ### just write duration and average of each activity, it needs t-test result
        write_duration('duration.csv', act_times, act_duration, cat = 'Duration')
        ### return the result of t_test
        return act_times, acts, act_duration

    if duration_time=="Start":
        act_start = {}
        act_start[0] = {}
        if len(group_members) > 1:
            for i in range(len(group_members)):
                if i != 0:
                    act_start[i] = {}
        result, act_times[0], acts[0], act_start[0] = ngram(filename, group_members[0], n=1,
                                              duration_time=duration_time)  # n is the length of sequence
        for k in range(len(group_members)):
            if k != 0:
                result, act_times[k], acts[k], act_start[k] = ngram(filename, group_members[k], n=1, duration_time=duration_time)  # n is the length of sequence
        ### just write duration and average of each activity, it needs t-test result
        write_duration('start.csv', act_times, act_start, cat = 'Start Time')
        ### return the result of t_test
        return act_times, acts, act_start


def ngram(filename, group_member, n=1, duration_time=False):
    #read 122traces_022717.csv
    attributes = []
    with open(filename, 'rU') as csvfile:
        reader = csv.reader(csvfile)
        l_n = 0
        for row in reader:
            if l_n == 0:
                list_label = row
            if l_n != 0:
                attributes.append(row)
            l_n += 1
    # countN for store the times of N-gram
    n_gram = []
    countN = {}
    activities_times = {}
    result = []
    activities = []

    if duration_time==False:
        # collect the activities of a patient
        activity_sequence, id_ = attr2act(attributes)
        # counting
        for num in group_member:
            if activity_sequence.has_key(num) == False:
                continue
            stri = activity_sequence[num]
            # count n_gram
            for i in range(len(stri) - n + 1):
                act = ''
                for j in range(n):
                    if j == 0:
                        act += stri[i + j]
                    else:
                        act = act + ', ' + stri[i + j]
                if countN.has_key(act):
                    countN[act] = countN[act] + 1
                else:
                    n_gram.append(act)
                    countN[act] = 1
        if n >= 1:
            for ng in n_gram:
                if countN[ng] > 0.2 * group_member.__len__():
                    result.append([countN[ng], ng])
                    activities_times[ng] = countN[ng]
                    activities.append(ng)
            # return result[list], the dictionary[dic] and the activities[list]
            return result, activities_times, activities
        else:
            raise ValueError('The n should be larger than or equal to 1!')

    if duration_time==True:
        countTime = {}
        # collect the activities of a patient
        activity_sequence, id_ = attr2act(attributes, type="durationTime")
        # counting
        for num in group_member:
            if activity_sequence.has_key(num) == False:
                continue
            stri = activity_sequence[num]
            # count n_gram
            for i in range(0, len(stri) - n + 1, 2):
                act = ''
                for j in range(n):
                    if j == 0:
                        act += stri[i + j]
                    else:
                        act = act + ', ' + str(stri[i + j])
                if countN.has_key(act):
                    countN[act] = countN[act] + 1
                    #countTime[act] = countTime[act].append(stri[i+1])
                    tmp = countTime[act]
                    tmp.append(stri[i+1])
                    countTime[act] = tmp
                else:
                    n_gram.append(act)
                    countN[act] = 1
                    countTime[act] = [stri[i+1]]

        if n >= 1:
            for ng in n_gram:
                result.append([countN[ng], ng])
                activities_times[ng] = countN[ng]
                activities.append(ng)
            return result, activities_times, activities, countTime
        else:
            raise ValueError('The n should be larger than or equal to 1!')

    if duration_time=="Start":
        countTime = {}
        # collect the activities of a patient
        activity_sequence, id_ = attr2act(attributes, type="startTime")
        # counting
        for num in group_member:
            if activity_sequence.has_key(num) == False:
                continue
            stri = activity_sequence[num]
            # count n_gram
            for i in range(0, len(stri) - n + 1, 2):
                act = ''
                for j in range(n):
                    if j == 0:
                        act += stri[i + j]
                    else:
                        act = act + ', ' + str(stri[i + j])
                if countN.has_key(act):
                    countN[act] = countN[act] + 1
                    # countTime[act] = countTime[act].append(stri[i+1])
                    tmp = countTime[act]
                    tmp.append(stri[i + 1])
                    countTime[act] = tmp
                else:
                    n_gram.append(act)
                    countN[act] = 1
                    countTime[act] = [stri[i + 1]]

        if n >= 1:
            for ng in n_gram:
                result.append([countN[ng], ng])
                activities_times[ng] = countN[ng]
                activities.append(ng)
            return result, activities_times, activities, countTime
        else:
            raise ValueError('The n should be larger than or equal to 1!')


def attr2act(attributes, type = "act"):
    #if wanna duration time, change type to other words
    if type == "act":
        activity_sequence = {}
        id_ = []
        for row in attributes:
            if activity_sequence.has_key(row[0]):
                list = activity_sequence[row[0]]
                list.append(row[1])
                activity_sequence[row[0]] = list
            else:
                id_.append(row[0])
                activity_sequence[row[0]] = [row[1]]
        return activity_sequence, id_

    if type == "durationTime":
        activity_sequence_time = {}
        id_ = []
        for row in attributes:
            #activity duration
            s1 = row[2]
            s2 = row[3]
            FMT = '%H:%M:%S'
            duration = datetime.strptime(s2, FMT) - datetime.strptime(s1, FMT)
            if activity_sequence_time.has_key(row[0]):
                list = activity_sequence_time[row[0]]
                list.append(row[1])
                list.append(duration.seconds)
                activity_sequence_time[row[0]] = list
            else:
                id_.append(row[0])
                activity_sequence_time[row[0]] = [row[1], duration.seconds]
        return activity_sequence_time, id_

    if type == "startTime":
        activity_sequence_time = {}
        id_ = []
        pt_arrival = ''
        for row in attributes:
            #activity duration
            if row[1]=='Pt arrival':
                pt_arrival=row[2]
            s1 = row[2]
            FMT = '%H:%M:%S'
            start_time = datetime.strptime(s1, FMT) - datetime.strptime(pt_arrival, FMT)
            if activity_sequence_time.has_key(row[0]):
                list = activity_sequence_time[row[0]]
                list.append(row[1])
                list.append(start_time.seconds)
                activity_sequence_time[row[0]] = list
            else:
                id_.append(row[0])
                activity_sequence_time[row[0]] = [row[1], start_time.seconds]
        return activity_sequence_time, id_


def significant_test(act_times, acts, group_members):
    clusters = len(acts)
    must_be_event_group = [[]for i in range(clusters)]
    # n-dimesional dictionaries
    maybe_event = {}
    maybe_event[0] = {}
    activities = [[]for i in range(clusters)]
    if len(group_members)>1:
        for i in range(len(group_members)):
            if i!=0:
                maybe_event[i] = {}

    # calculate act times total
    act_times_total = {}
    for i in range(clusters):
        for a in acts[i]:
            if act_times_total.has_key(a):
                act_times_total[a] = act_times_total[a] + act_times[i][a]
            else:
                act_times_total[a] = act_times[i][a]

    for i in range(clusters):
        for a in acts[i]:
            if act_times_total[a] != act_times[i][a] \
                    and act_times_total[a]-act_times[i][a]< 122-len(group_members[i]) \
                    and act_times[i][a] < len(group_members[i]):
                p = p_value(zscore_calculator(act_times[i][a], len(group_members[i]),
                                              act_times_total[a]-act_times[i][a], 122-len(group_members[i])))
                if p < 0.05:
                    maybe_event[i][a] = p
                    activities[i].append(a)
    write_pvalue('pvalue.csv', maybe_event, activities)
    return maybe_event, activities


def zscore_calculator(x1, y1, x2, y2):
    x1 = float(x1)
    x2 = float(x2)
    y1 = float(y1)
    y2 = float(y2)
    return (x1 / y1 - x2 / y2) / math.sqrt((x1 + x2) / (y1 + y2) * (1 - (x1 + x2) / (y1 + y2)) * (1 / y1 + 1 / y2))


def p_value(z):
    return st.norm.cdf(z)

### need to be edited
def calculate_total(act_duration, i):
    dict = {}
    for n in range(len(act_duration)):
        if n != i:
            for act in act_duration[n]:
                if dict.has_key(act):
                    dict[act] = dict[act] + act_duration[n][act]
                else:
                    dict[act] = act_duration[n][act]
    return dict

# clean the origin file and write
def write_ngram(filename, data, groupmember, n = 1, type = "new"):
    if type == "new":
        with open(filename, 'wb+') as file:
            file.write('This group include: ')
            for item in groupmember:
                file.write('"' + item + '" ')
            file.writelines('\nn = 1\n')
            wr = csv.writer(file, quoting=csv.QUOTE_ALL)
            for row in data:
                wr.writerow(row)
    elif type == "new_group":
        with open(filename, 'ab') as file:
            file.write('\nThis group include: ')
            for item in groupmember:
                file.write('"' + item + '" ')
            file.writelines('\nn = 1\n')
            wr = csv.writer(file, quoting=csv.QUOTE_ALL)
            for row in data:
                wr.writerow(row)
    elif type == "append":
        with open(filename, 'ab') as file:
            file.writelines('\n' + 'n = ' + str(n) + '\n')
            wr = csv.writer(file, quoting=csv.QUOTE_ALL)
            for row in data:
                wr.writerow(row)


def write_attributes(filename, attributes, group_member, n, list_label):
    result = [[] for i in range(len(group_member))]
    result_no_ids = [[] for i in range(len(group_member))]
    with open(filename, 'wb+') as file:
        for attr in attributes:
            if attr[0] in group_member[0]:
                for i in attr:
                    file.write(str(i)+',')
                file.write('\n')
                result[0].append(attr)
                result_no_ids[0].append(attr[2:])
        for k in range(n):
            if k!=0:
                file.write('\n')
                for attr in attributes:
                    if attr[0] in group_member[k]:
                        for i in attr:
                            file.write(str(i) + ',')
                        file.write('\n')
                        result[k].append(attr)
                        result_no_ids[k].append(attr[2:])
    return result, result_no_ids


def write_pvalue(filename, maybe_event, activities):
    with open(filename, 'wb+') as file:
        for i in range(len(activities)):
            file.write('Group ' + str(i) + ' sequential pattern that less than 0.05:\n')
            for a in activities[i]:
                #file.write(str(maybe_event[i][a])+ ' ' +a + '\n')
                file.write(str(format(maybe_event[i][a], '.3f'))+' ' +a + '\n')
            file.write('\n\n')


def write_duration(filename, act_times, act_duration, cat=""):
    with open(filename, 'wb+') as file:
        file.writelines(cat + '\n')
        for i in range(len(act_duration)):
            act_duration_total = calculate_total(act_duration, i)
            transfer = []
            file.write('Group ' + str(i) + ' activities\' duration:\n')
            for a in act_duration[i]:
                avg, std = calculate(act_duration[i][a])
                t = -1
                p = -1
                if std != 0:
                    array1 = np.asarray(act_duration[i][a])
                    array2 = np.asarray(act_duration_total[a])
                    t, p = stats.ttest_ind(array1, array2)
                #transfer.append([avg, std, a, len(act_duration[i][a]), t, p])
                transfer.append([p, avg, std, a, len(act_duration[i][a]), t])
            transfer.sort()
            for act in transfer:
                #file.write(act[2].ljust(25)+ 'times: '+ str(act_times[i][act[2]]).ljust(5) + 'avg:' + (str(act[0])+'s').ljust(16)
                #           + ' std: ' + str(act[1]).ljust(15) + ' p-value: ' + str(act[5]) + '\n')
                file.write(act[3].ljust(25)+' p-value: '+ str(format(act[0], '.3f')).ljust(10)+ 'avg:' + (
                str(format(act[1], '.3f')) + 's').ljust(10) + ' std: ' + str(format(act[2], '.3f')) +  '\n')
            file.write('\n')


def calculate(time):
    a = np.array(time)
    return np.average(a), np.std(a)


def t_test(avg1, avg2, std1, std2, n1, n2):
    if math.sqrt( pow(std1,2)/n1 + pow(std2,2)/n2 ) == 0:
        return -1
    else:
        return p_value(abs((avg1-avg2)/math.sqrt( pow(std1,2)/n1 + pow(std2,2)/n2 )))


def show_kmeans(kmeans, reduced_data):
    plt.figure(1)
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')
    # plot all point in black
    # plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

    # plot points in different colors
    for i in range(len(kmeans.labels_)):
        if (kmeans.labels_[i] == 0):
            plt.plot(reduced_data[i, 0], reduced_data[i, 1], 'gx')
        if (kmeans.labels_[i] == 1):
            plt.plot(reduced_data[i, 0], reduced_data[i, 1], 'rD')
        if (kmeans.labels_[i] == 2):
            plt.plot(reduced_data[i, 0], reduced_data[i, 1], 'k+')
        if (kmeans.labels_[i] == 3):
            plt.plot(reduced_data[i, 0], reduced_data[i, 1], 'c*')

    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering of '+ str(len(centroids))+ ' groups. (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    # plt.xticks(())
    # plt.yticks(())

    #plt.show()


def show_kmedoid(M, C, reduced_data):
    plt.figure(1)
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1

    # Put the result into a color plot
    plt.figure(1)
    plt.clf()

    for i in range(len(C)):
        if i==0:
            for i in C[0]:
                plt.plot(reduced_data[i, 0], reduced_data[i, 1], 'gx')
        if i==1:
            for i in C[1]:
                plt.plot(reduced_data[i, 0], reduced_data[i, 1], 'rD')
        if i==2:
            for i in C[2]:
                plt.plot(reduced_data[i, 0], reduced_data[i, 1], 'k+')
        if i == 3:
            for i in C[3]:
                plt.plot(reduced_data[i, 0], reduced_data[i, 1], 'c*')
    # Plot the centroids as a white X
    plt.scatter(reduced_data[M[:], 0], reduced_data[M[:], 1],
                marker='x', s=169, linewidths=3,
                color='c', zorder=10)
    plt.title('K-medoids clustering of '+ str(len(M))+ ' groups. (PCA-reduced data)\n'
              'Centroids are marked with cyan cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)


def cal_avg(data):
    result = []
    total = [0 for i in range(len(data[0][0]))]
    for da in data:
        oneline = [0 for i in range(len(da[0]))]
        max_ele = [0 for i in range(len(da[0]))]
        for i in range(len(da)):
            for j in range(len(da[i])):
                oneline[j] = oneline[j] + int(da[i][j])
                total[j] = total[j] + int(da[i][j])
                if int(da[i][j]) > max_ele[j]:
                    max_ele[j] = int(da[i][j])
        for i in range(len(oneline)):
            if oneline[i]!=0:
                oneline[i] = oneline[i]/float(max_ele[i])/float(len(da))
        result.append(oneline)
    for i in range(len(total)):
        total[i] = total[i] / float(123)
    return result, total


def deduct_dimension(data):
    array1 = [[] for i in range(len(data))]
    array2 = [[] for i in range(len(data[0][0]))]
    for i in range(len(data)):
        for j in range(len(data[i][0])):
            array1[i].append([int(x[j]) for x in data[i]])
    for i in range(len(array1[0])):
        for j in range(len(array1)):
            array2[i] = array2[i] + array1[j][i]
    result = [[] for i in range(len(data))]
    for i in range(len(array1)):
        for j in range(len(array1[0])):
            ar1 = np.asarray(array1[i][j])
            ar2 = np.asarray(array2[j])
            t, p = stats.ttest_ind(ar1, ar2)
            result[i].append(p)
    diff = []
    for i in range(len(result)):
        for j in range(len(result[0])):
            if result[i][j]< 0.05:
                diff.append(j)
    return diff


def radar(cat, data):
    diff = deduct_dimension(data)
    colors = ['b', 'r', 'y', 'g', 'm']
    colors = colors[:len(data)]
    each_group, total = cal_avg(data)

    new_cat = []
    new_each_group = [[] for i in range(len(each_group))]
    new_total = []
    for i in range(len(cat)):
        if i in diff:
            new_cat.append(cat[i])
            new_total.append(total[i])
    for i in range(len(each_group)):
        for j in range(len(each_group[i])):
            if j in diff:
                new_each_group[i].append(each_group[i][j])
    #total = new_total
    each_group = new_each_group
    cat = new_cat
    N = len(cat)
    for values, color, ii in zip(each_group, colors, range(len(data))):
        plt.figure(ii+2, figsize=(5,5))

        x_as = [n / float(N) * 2 * pi for n in range(N)]

        # Because our chart will be circular we need to append a copy of the first
        # value of each list at the end of each list with data
        values += values[:1]
        x_as += x_as[:1]

        # Set color of axes
        plt.rc('axes', linewidth=0.5, edgecolor="#888888")

        # Create polar plot
        ax = plt.subplot(111, polar=True)

        # Set clockwise rotation. That is:
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        # Set color and linestyle of grid
        ax.xaxis.grid(True, color="#888888", linestyle='solid', linewidth=0.5)
        ax.yaxis.grid(True, color="#888888", linestyle='solid', linewidth=0.5)

        # Set number of radial axes and remove labels
        plt.xticks(x_as[:-1], [])

        # Set yticks
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1], ["0.2", "0.4", "0.6", "0.8", "1"])

        # Plot data
        ax.plot(x_as, values, color, linewidth=1.5, linestyle='solid', zorder=3)

        # Fill area
        ax.fill(x_as, values, color, alpha=0.3)

        # Set axes limits
        plt.ylim(0, 1)

        # Draw ytick labels to make sure they fit properly
        for i in range(N):
            angle_rad = i / float(N) * 2 * pi

            if angle_rad == 0:
                ha, distance_ax = "center", 1
            elif 0 < angle_rad < pi:
                ha, distance_ax = "left", 1
            elif angle_rad == pi:
                ha, distance_ax = "center", 1
            else:
                ha, distance_ax = "right", 1

            ax.text(angle_rad,  0.05  + distance_ax, cat[i], size=10, horizontalalignment=ha, verticalalignment="center")

        # Show polar plot
        plt.title("Attributes of each Group", fontsize = 20)
        font = {'family': 'serif',
            'color':  colors[ii],
            'weight': 'normal',
            'size': 16}
        plt.text(0.5, 1.15, 'group '+ str(ii), fontdict = font)
        #plt.show()


def random_pick_up(members, number_each_group):
    length = len(members)
    array = [i for i in range(length)]
    rand = []
    while length > 0:
        i = random.randint(0,length-1)
        rand.append(array[i])
        array[i], array[length-1] = array[length-1], array[i]
        length = length-1
    result_random = [[] for i in range(len(rand) / number_each_group)]
    for i in range(len(rand) / number_each_group):
        for j in range(number_each_group):
            result_random[i].append(members[rand[i*number_each_group + j]])

    return result_random


def silhouette(X):
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9]

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors)

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1],
                    marker='o', c="white", alpha=1, s=200)

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
        plt.show()

if __name__ == "__main__":
    numberOfClusters = 3

    ### read attributes without weights
    #attributes, ids, list_label, attributes_all = read_attributes('attributes_05.15.2017.csv')

    ### read attributes with weights
    #AGE_Group	Male_1	Transfer	Stat	Attending	Blunt	Penetrating	Animal Bite	Burn	Blunt/No Injury	Non-critical admission
    # Critical Admission	Discharged	ETA_Now	Weekend	Intubation	Daytime	GCS>13	ISS Group	AIS_HEAD_NECK	AIS_FACE	AIS_CHEST	AIS_ABD_PELVIC	AIS_EXTREMITIE	AIS_EXTERNAL	Maximum AIS
    weights = [2, 5, 3, 1, 3, 1, 176, 4, 4, 164, 2, 1,
               75, 3, 2, 5, 3, 4, 1, 1, 1, 3, 1, 1, 4, 3]
    attributes, ids, list_label, attributes_all = read_attributes_weight('attributes_05.15.2017.csv', weights)




    ### for kmedoid and kmeans, change by commenting each other
    group_members, kmeans, reduced_data = kmeans(attributes, ids, n = numberOfClusters)
    #group_members, M, C, reduced_data = kmedoid(attributes, ids, n= numberOfClusters)


    ### write the attributes
    result, result_no_id = write_attributes('attributes.csv', attributes_all, group_members, n = numberOfClusters, list_label=list_label)

    ### three methods to attract n-gram. Each method will write the results to different files:
    # 'without_duration.csv' , 'duration.csv' , 'start.csv'
    #  change the file name in function: do_ngram
    #
    # duration_time = False (default)
    # duration_time = True   # include duration time
    # duration_time = 'Start' # include start time
    do_ngram(filename='122traces_02.27.17.csv', group_members = group_members)
    do_ngram(filename='122traces_02.27.17.csv', group_members=group_members,
                                                    duration_time=True)
    act_times, acts, act_durationOrStart = do_ngram(filename='122traces_02.27.17.csv',
                                                    group_members = group_members, duration_time='Start')

    ### do significant test and write sequential pattern whose p-value is less than 0.05 to 'pvalue.csv'
    maybe_event, activities = significant_test(act_times, acts, group_members)

    ### plot radar chart according to the attributes
    radar(list_label[2:], result_no_id)
    plt.show()


    silhouette(reduced_data)
    '''
    ### random pich 123 / 3 = 41 groups of patients
    number_each_group = 3
    random_pick_up(ids, number_each_group)
    '''
