from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import csv
import math
import scipy.stats as st
import matplotlib.pyplot as plt
import kmedoids


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
    # return kmeans result and ids of patients for tracing
    return group_members, M, C, reduced_data


def do_ngram(filename, group_members):
    #define multi-dimension dictionaries
    act_times = {}
    act_times[0] = {}
    acts = [[] for i in range(len(group_members))]
    if len(group_members)>1:
        for i in range(len(group_members)):
            if i!=0:
                act_times[i] = {}
    result, act_times[0], acts[0] = ngram(filename, group_members[0], n = 1) #n is the length of sequence
    result.sort(reverse = True)
    write_ngram('output.csv', result, group_members[0], type="new")
    for i in range(1000):
        result, temp, temp2 = ngram(filename, group_members[0], i+2)
        result.sort(reverse=True)
        if (len(result) == 0):
            break
        write_ngram('output.csv', result, group_members[0], n = i + 2, type="append")
        act_times[0].update(temp)
        acts[0] = acts[0] + temp2

    for k in range(len(group_members)):
        if k!=0:
            result, act_times[k], acts[k] = ngram(filename, group_members[k], n=1)  # n is the length of sequence
            result.sort(reverse=True)
            write_ngram('output.csv', result, group_members[k], type="new_group")
            for i in range(1000):
                result, temp, temp2 = ngram(filename, group_members[k], i + 2)
                result.sort(reverse=True)
                if (len(result) == 0):
                    break
                write_ngram('output.csv', result, group_members[k], n=i+2, type="append")
                act_times[k].update(temp)
                acts[k] = acts[k] + temp2
    return act_times, acts


def ngram(filename, group_member, n=1):
    #read 122traces_022717.csv
    attributes = []
    list_label = []
    with open(filename, 'rU') as csvfile:
        reader = csv.reader(csvfile)
        l_n = 0
        for row in reader:
            if l_n == 0:
                list_label = row
            if l_n != 0:
                attributes.append(row)
            l_n += 1

    cluster = len(group_member)
    # countN for store the times of N-gram
    n_gram = []
    countN = {}
    activities_times = {}
    result = []
    activities = []

    # collect the activities of a patient
    activity_sequence, id_ = attr2act(attributes)

    # counting
    for num in group_member:
        if activity_sequence.has_key(num) == False:
            continue
        str = activity_sequence[num]
        # count n_gram
        for i in range(len(str) - n + 1):
            act = ''
            for j in range(n):
                if j == 0:
                    act += str[i + j]
                else:
                    act = act + ', ' + str[i + j]
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


def attr2act(attributes, type = "act"):
    #if wanna duration time, change type to other words
    activity_sequence = {}
    id_ = []
    for row in attributes:
        ''' not used yet
        #activity duration
        s1 = row[2]
        s2 = row[3]
        FMT = '%H:%M:%S'
        duration = datetime.strptime(s2, FMT) - datetime.strptime(s1, FMT)
        '''
        if activity_sequence.has_key(row[0]):
            list = activity_sequence[row[0]]
            list.append(row[1])
            activity_sequence[row[0]] = list
        else:
            id_.append(row[0])
            activity_sequence[row[0]] = [row[1]]
    return activity_sequence, id_


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
    return maybe_event, activities


def zscore_calculator(x1, y1, x2, y2):
    x1 = float(x1)
    x2 = float(x2)
    y1 = float(y1)
    y2 = float(y2)
    return (x1 / y1 - x2 / y2) / math.sqrt((x1 + x2) / (y1 + y2) * (1 - (x1 + x2) / (y1 + y2)) * (1 / y1 + 1 / y2))


def p_value(z):
    return st.norm.cdf(z)


# clean the origin file and write
def write_ngram(filename, data, groupmember, n = 1, type = "new"):
    if type == "new":
        with open(filename, 'wb') as file:
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
    with open(filename, 'wb') as file:
        for attr in attributes:
            if attr[0] in group_member[0]:
                for i in attr:
                    file.write(str(i)+',')
                file.write('\n')
        for k in range(n):
            if k!=0:
                file.write('\n')
                for attr in attributes:
                    if attr[0] in group_member[k]:
                        for i in attr:
                            file.write(str(i) + ',')
                        file.write('\n')


def write_pvalue(filename, maybe_event, activities):
    with open(filename, 'wb') as file:
        for i in range(len(activities)):
            file.write('Group ' + str(i) + ' sequential pattern that less than 0.05:\n')
            for a in activities[i]:
                file.write(str(maybe_event[i][a])+ ' ' +a + '\n')
            file.write('\n\n')


def show_kmeans(kmeans, reduced_data):
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
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    # plt.xticks(())
    # plt.yticks(())
    plt.show()


def show_kmedoid(M, C, reduced_data):
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
    plt.title('K-medoids clustering on the digits dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    # plt.xticks(())
    # plt.yticks(())
    plt.show()


if __name__ == "__main__":
    numberOfClusters = 3
    attributes, ids, list_label, attributes_complete = read_attributes('attributes_05.15.2017.csv')

    #group_members, kmeans, reduced_data = kmeans(attributes, ids, n = numberOfClusters)
    group_members, M, C, reduced_data = kmedoid(attributes, ids, n= numberOfClusters)
    #show_kmeans(kmeans, reduced_data)
    show_kmedoid(M, C, reduced_data)

    write_attributes('attributes.csv', attributes_complete, group_members, n = numberOfClusters, list_label=list_label)

    act_times, acts = do_ngram(filename='122traces_02.27.17.csv', group_members = group_members)
    maybe_event, activities = significant_test(act_times, acts, group_members)
    write_pvalue('pvalue.csv', maybe_event, activities)
