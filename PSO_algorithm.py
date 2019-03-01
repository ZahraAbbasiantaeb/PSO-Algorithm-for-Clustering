import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from question4.individual import individual


class PSO_algorithm:

    cluster_Length = 0

    num_of_cluster = 0

    data = []

    individuals = []

    individual_count = 0

    iterations =  0

    fitness_avg = []

    best_pred_lables = []

    min_index = []

    max_index = []

    actual_lables=[]


    def __init__(self, data, num_of_cluster, cluster_Length, individual_count, iterations, min_index, max_index, actual_lables):

        self.actual_lables = actual_lables
        self.min_index = min_index
        self.max_index = max_index
        self.iterations = iterations
        self.individual_count = individual_count
        self.data = data
        self.num_of_cluster = num_of_cluster
        self.cluster_Length = cluster_Length


    def run(self):

        for j in range(0, self.iterations):
            min_cost = 0
            i = 0
            index = 0
            new_cost = []
            totalCost = 0
            print("iter is: "+str(j))

            for individual in self.individuals:

                cost, lables_pred = self.findError(individual)
                totalCost += cost


                if (cost < min_cost or i==0):
                    min_cost = cost
                    index = i
                    self.best_pred_lables = lables_pred

                individual.updateBest(cost)
                new_cost.append(i)
                i += 1

                print(normalized_mutual_info_score(self.actual_lables, lables_pred))
                print(cost)
                print("*****")
            for individual in self.individuals:

                individual.update_GBest(self.individuals[index].pos)
                individual.update_position()

            self.fitness_avg.append((totalCost/self.individual_count))
        plt.plot(self.fitness_avg)
        plt.xlabel("iteration")
        plt.ylabel("Error")
        plt.show()
        print(self.fitness_avg)
        print(self.best_pred_lables)

        return self.best_pred_lables


    def initialPopulation(self, centroids):

        p=0;

        position = np.zeros(self.cluster_Length * self.num_of_cluster)

        for elem in centroids:
            row = 0

            for i in range(0, len(elem)):
                position[p] = elem[row]
                row+=1
                p+=1

        velocity = [random.uniform(0,1) for x in range(0, self.cluster_Length * self.num_of_cluster)]

        self.individuals.append(individual(position, position, velocity))

        for x in range(1, self.individual_count):

            position = np.zeros(self.cluster_Length * self.num_of_cluster)

            for i in range(0, self.cluster_Length * self.num_of_cluster):
                position[i] = random.uniform(self.min_index[i%self.num_of_cluster],self.max_index[i%self.num_of_cluster])

            velocity = [random.uniform(0, 1) for z in range(0, self.cluster_Length * self.num_of_cluster)]

            self.individuals.append(individual(position, position, velocity))

        totalCost=0

        i=0

        min_cost=0

        for element in self.individuals:

            cost, lables_pred = self.findError(element)
            totalCost += cost

            if (cost < min_cost or i==0):
                min_cost = cost
                index = i
                self.best_pred_lables = lables_pred

                element.setCost(cost)
            i += 1

        for element in self.individuals:
            element.update_GBest(self.individuals[index].pos)


        return


    def findError(self, individual):

        cents = []

        for i in range(0, self.num_of_cluster):

            cent = np.zeros(self.cluster_Length)

            for j in range(0, self.cluster_Length):

                cent[j] = individual.pos[i * self.cluster_Length + j]

            cents.append(cent)

        sum = np.zeros(self.num_of_cluster)
        count = np.zeros(self.num_of_cluster)

        labels_pred = []
        length = len(self.data)

        for z in range(0, length):

            min_distance = 0
            cluster_index = 0

            for j in range(0, self.num_of_cluster):

                dist = self.getDistance(self.data[z], cents[j])
                if (dist < min_distance or j==0 ):

                    min_distance = dist
                    cluster_index = j

            sum[cluster_index] += min_distance
            count[cluster_index] += 1

            labels_pred.append(cluster_index)

        for i in range(0, self.num_of_cluster):

            if(count[i]>0):
                sum[i]/=count[i]

        print(set(labels_pred))

        return (np.sum(sum)/self.num_of_cluster), labels_pred


    def getDistance(self, x, y):

        length = len(x)
        sum = 0

        for i in range(0, length):
            if((x[i] - y[i]) ** 2< np.math.inf):
                sum += (x[i] - y[i]) ** 2

        return sum