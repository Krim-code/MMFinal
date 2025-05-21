import math

import numpy as np
import calc_dist



all_type = {'евклидово':calc_dist.euclidean,
                         'манхэттенское':calc_dist.manhattan,
                         'взвешенное евклидово':calc_dist.euclidean_w,
                         'степенное':calc_dist.stepen
                         }


class Cluster:
    def __init__(self, objects, distance=0.0, name=None):
        self.name = name
        self.distance = distance
        self.objects = objects

    def to_list(self):
        return f'{self.name}, {self.objects}, {self.distance}'
    # Метод нахождения расстояния между кластерами (мин/макс)
    def get_distance(self, cluster, type_calc, is_min_distance=True):
        base_dist = all_type[type_calc](self.objects[0], cluster.objects[0])
        for obj in self.objects:
            for other_object in cluster.objects:
                distance = all_type[type_calc](obj, other_object)
                if is_min_distance:
                    if distance < base_dist:
                        base_dist = distance
                else:
                    if distance > base_dist:
                        base_dist = distance
        return base_dist

    def get_distance_centroid(self, cluster, type_calc):

        sum_cl1 = calc_dist.calc_centroid(self.objects)
        sum_cl2 = calc_dist.calc_centroid(cluster.objects)

        # print(sum_cl1, sum_cl2)

        return all_type[type_calc](sum_cl1, sum_cl2)

    def get_distance_average(self, cluster, type_calc):
        sum_ = 0
        for obj in self.objects:
            for other_object in cluster.objects:
                sum_ += all_type[type_calc](obj, other_object)


        return sum_ / (len(self.objects) + len(cluster.objects))



class CalcSimilar():
    def __init__(self, df, max_len_cluster, type_calc):
        self.df = df
        self.max_len_cluster = max_len_cluster
        self.type_calc = type_calc
        self.func = all_type[type_calc]

    def calculate(self, t):
        all_type = {'одиночной связи': self.near_neighbor,
                    'полных связей': self.dist_neighbor,
                    'центроидный': self.centroid,
                    'средних связей': self.medium_con,
                    'Уорда': self.uorda}

        return all_type[t]()

    def near_neighbor(self):
        '''
        –	метод одиночной связи
        ближайшего соседа
        '''
        clusters_answer = []
        clusters = [Cluster([self.df[i]], 0, i) for i in range(len(self.df))]
        clusters_merge_matrix = []
        cluster_name = len(clusters)

        while len(clusters) > 1:
            min_distance = float("inf")
            length = len(clusters)
            for i in range(length):
                for j in range(i + 1, length):
                    dist = clusters[i].get_distance(clusters[j], self.type_calc)
                    if dist < min_distance and (len(clusters[i].objects) + len(clusters[j].objects) <= self.max_len_cluster):
                        min_distance = dist
                        closest_clusters = (i, j)
            if min_distance == float('inf'):
                break
            first_cluster = clusters[closest_clusters[0]]
            second_cluster = clusters[closest_clusters[1]]
            new_cluster = Cluster(first_cluster.objects + second_cluster.objects,
                                  min_distance, cluster_name)

            clusters_merge_matrix.append([first_cluster.name, second_cluster.name,
                                          new_cluster.distance, len(new_cluster.objects)])
            cluster_name += 1

            del clusters[closest_clusters[1]]
            del clusters[closest_clusters[0]]
            clusters.append(new_cluster)
            for cluster in clusters:
                if len(cluster.objects) == self.max_len_cluster:
                    clusters_answer.append(clusters[clusters.index(cluster)])
                    del clusters[clusters.index(cluster)]
            # print(len(clusters_answer))


        for cluster in clusters:
            clusters_answer.append(clusters[clusters.index(cluster)])
        # print(len(clusters_answer))

        clusters_result = []

        for cluster in clusters_answer:
            cluster_ = []
            for i, cl in enumerate(self.df):
                for ob in cluster.objects:
                    if cl == ob:
                        cluster_.append(i)
            clusters_result.append(cluster_)



        # for cluster in clusters_answer:
        #     print(cluster.to_list())
        return clusters_result

    def dist_neighbor(self):
        '''
        –	метод полных связей
        дальнего соседа
        '''
        clusters_answer = []
        clusters = [Cluster([self.df[i]], 0, i) for i in range(len(self.df))]
        clusters_merge_matrix = []
        cluster_name = len(clusters)

        while len(clusters) > 1:
            min_distance = float("inf")
            length = len(clusters)
            for i in range(length):
                for j in range(i + 1, length):
                    dist = clusters[i].get_distance(clusters[j], self.type_calc, False)
                    if (dist < min_distance and (
                            len(clusters[i].objects) + len(clusters[j].objects) <= self.max_len_cluster)):
                        min_distance = dist
                        closest_clusters = (i, j)
            if min_distance == float('inf'):
                break
            first_cluster = clusters[closest_clusters[0]]
            second_cluster = clusters[closest_clusters[1]]
            new_cluster = Cluster(first_cluster.objects + second_cluster.objects,
                                  min_distance, cluster_name)

            clusters_merge_matrix.append([first_cluster.name, second_cluster.name,
                                          new_cluster.distance, len(new_cluster.objects)])
            cluster_name += 1

            del clusters[closest_clusters[1]]
            del clusters[closest_clusters[0]]
            clusters.append(new_cluster)
            for cluster in clusters:
                if len(cluster.objects) == self.max_len_cluster:
                    clusters_answer.append(clusters[clusters.index(cluster)])
                    del clusters[clusters.index(cluster)]
            # print(len(clusters_answer))

        for cluster in clusters:
            clusters_answer.append(clusters[clusters.index(cluster)])
        # print(len(clusters_answer))

        clusters_result = []

        for cluster in clusters_answer:
            cluster_ = []
            for i, cl in enumerate(self.df):
                for ob in cluster.objects:
                    if cl == ob:
                        cluster_.append(i)
            clusters_result.append(cluster_)

        # for cluster in clusters_answer:
        #     print(cluster.to_list())
        return clusters_result

    def centroid(self):
        '''
        –	центроидный метод
        центроид
        '''
        clusters_answer = []
        clusters = [Cluster([self.df[i]], 0, i) for i in range(len(self.df))]
        clusters_merge_matrix = []
        cluster_name = len(clusters)

        while len(clusters) > 1:
            min_distance = float("inf")
            length = len(clusters)
            for i in range(length):
                for j in range(i + 1, length):
                    dist = clusters[i].get_distance_centroid(clusters[j], self.type_calc)
                    if dist < min_distance and (
                            len(clusters[i].objects) + len(clusters[j].objects) <= self.max_len_cluster):
                        min_distance = dist
                        closest_clusters = (i, j)
            if min_distance == float('inf'):
                break
            first_cluster = clusters[closest_clusters[0]]
            second_cluster = clusters[closest_clusters[1]]
            new_cluster = Cluster(first_cluster.objects + second_cluster.objects,
                                  min_distance, cluster_name)

            clusters_merge_matrix.append([first_cluster.name, second_cluster.name,
                                          new_cluster.distance, len(new_cluster.objects)])
            cluster_name += 1

            del clusters[closest_clusters[1]]
            del clusters[closest_clusters[0]]
            clusters.append(new_cluster)
            for cluster in clusters:
                if len(cluster.objects) == self.max_len_cluster:
                    clusters_answer.append(clusters[clusters.index(cluster)])
                    del clusters[clusters.index(cluster)]
            # print(len(clusters_answer))

        for cluster in clusters:
            clusters_answer.append(clusters[clusters.index(cluster)])
        # print(len(clusters_answer))

        clusters_result = []

        for cluster in clusters_answer:
            cluster_ = []
            for i, cl in enumerate(self.df):
                for ob in cluster.objects:
                    if cl == ob:
                        cluster_.append(i)
            clusters_result.append(cluster_)

        # for cluster in clusters_answer:
        #     print(cluster.to_list())
        return clusters_result

    def medium_con(self):
        '''
        –	метод средней связи
        средней связи
        '''
        clusters_answer = []
        clusters = [Cluster([self.df[i]], 0, i) for i in range(len(self.df))]
        clusters_merge_matrix = []
        cluster_name = len(clusters)

        while len(clusters) > 1:
            min_distance = float("inf")
            length = len(clusters)
            for i in range(length):
                for j in range(i + 1, length):
                    dist = clusters[i].get_distance_average(clusters[j], self.type_calc)
                    if dist < min_distance and (
                            len(clusters[i].objects) + len(clusters[j].objects) <= self.max_len_cluster):
                        min_distance = dist
                        closest_clusters = (i, j)
            if min_distance == float('inf'):
                break
            first_cluster = clusters[closest_clusters[0]]
            second_cluster = clusters[closest_clusters[1]]
            new_cluster = Cluster(first_cluster.objects + second_cluster.objects,
                                  min_distance, cluster_name)

            clusters_merge_matrix.append([first_cluster.name, second_cluster.name,
                                          new_cluster.distance, len(new_cluster.objects)])
            cluster_name += 1

            del clusters[closest_clusters[1]]
            del clusters[closest_clusters[0]]
            clusters.append(new_cluster)
            for cluster in clusters:
                if len(cluster.objects) == self.max_len_cluster:
                    clusters_answer.append(clusters[clusters.index(cluster)])
                    del clusters[clusters.index(cluster)]
            # print(len(clusters_answer))

        for cluster in clusters:
            clusters_answer.append(clusters[clusters.index(cluster)])
        # print(len(clusters_answer))

        clusters_result = []

        for cluster in clusters_answer:
            cluster_ = []
            for i, cl in enumerate(self.df):
                for ob in cluster.objects:
                    if cl == ob:
                        cluster_.append(i)
            clusters_result.append(cluster_)

        # for cluster in clusters_answer:
        #     print(cluster.to_list())
        return clusters_result



    def ward_distance(self, c1, c2):
        n1, n2 = len(c1), len(c2)
        c1_mean, c2_mean = np.mean(c1, axis=0), np.mean(c2, axis=0)
        dist = all_type[self.type_calc](c1_mean, c2_mean)

        return (n1 * n2) / (n1 + n2) * dist
    def update_labels(self, labels, min_cdist_idxs):
        # assign a cluster number to labels
        labels[labels == min_cdist_idxs[1]] = min_cdist_idxs[0]
        labels[labels > min_cdist_idxs[1]] -= 1

        return labels

    def uorda(self):
        '''
        –	метод Уорда
        Уорда
        '''
        labels = np.arange(len(self.df))
        clusters = [[x] for x in self.df]

        while len(clusters) > 1:
            min_cdist, min_cdist_idxs = np.inf, []

            for i in range(len(clusters) - 1):
                for j in range(i + 1, len(clusters)):
                    cdist = self.ward_distance(clusters[i], clusters[j])
                    if cdist < min_cdist and len(clusters[i])+len(clusters[j]) <=self.max_len_cluster:
                        min_cdist = cdist
                        min_cdist_idxs = (i, j)

            if min_cdist == np.inf:
                break
            labels = self.update_labels(labels, min_cdist_idxs)
            clusters[min_cdist_idxs[0]].extend(clusters.pop(min_cdist_idxs[1]))

        answer = [[] for i in range(max(labels)+1)]
        for i, val in enumerate(labels):
            answer[val].append(i)
        return answer
            # hierarchical_cluster = AgglomerativeClustering(n_clusters=len(self.df)//self.max_len_cluster, affinity='euclidean', linkage='ward')
            # labels = hierarchical_cluster.fit_predict(self.df)
            # print(labels)
            # answer = [[] for i in range(len(self.df)//self.max_len_cluster)]
            # for i, val in enumerate(labels):
            #     answer[val].append(i)
            # print(answer)
            #

