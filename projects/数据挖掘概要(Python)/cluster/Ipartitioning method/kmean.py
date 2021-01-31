import math
import random 


def getMedian(alist):
    """get median of list"""
    tmp = list(alist)
    tmp.sort()
    alen = len(tmp)
    if (alen % 2) == 1:
        return tmp[alen // 2]
    else:
        return (tmp[alen // 2] + tmp[(alen // 2) - 1]) / 2
    

def normalizeColumn(column):
    """normalize the values of a column using Modified Standard Score
    that is (each value - median) / (absolute standard deviation)"""
    median = getMedian(column)
    asd = sum([abs(x - median) for x in column]) / len(column)
    result = [(x - median) / asd for x in column]
    return result


class kClusterer:
    
    def __init__(self, filename, k):
        """ k is the number of clusters to make
        This init method:
           1. reads the data from the file named filename
           2. stores that data by column in self.data
           3. normalizes the data using Modified Standard Score
           4. randomly selects the initial centroids
           5. assigns points to clusters associated with those centroids
        """
        file = open(filename)
        self.data = {}
        self.k = k
        self.counter = 0
        self.iterationNumber = 0
        # used to keep track of % of points that change cluster membership
        # in an iteration
        self.pointsChanged = 0
        # Sum of Squared Error
        self.sse = 0
        #
        # read data from file
        #
        lines = file.readlines()
        file.close()
        header = lines[0].split(',')
        self.cols = len(header)
        self.data = [[] for i in range(len(header))]
        # we are storing the data by column.
        # For example, self.data[0] is the data from column 0.
        # self.data[0][10] is the column 0 value of item 10.
        for line in lines[1:]:
            cells = line.split(',')
            toggle = 0
            for cell in range(self.cols):
                if toggle == 0:
                    self.data[cell].append(cells[cell])
                    toggle = 1
                else:
                    self.data[cell].append(float(cells[cell]))
                    
        self.datasize = len(self.data[1])
        self.memberOf = [-1 for x in range(len(self.data[1]))]
        #
        # now normalize number columns
        #
        for i in range(1, self.cols):
                self.data[i] = normalizeColumn(self.data[i])

        # select random centroids from existing points
        random.seed()
        self.centroids = [[self.data[i][r]  for i in range(1, len(self.data))]
                           for r in random.sample(range(len(self.data[0])),
                                                 self.k)]
        self.assignPointsToCluster()

            

    def updateCentroids(self):
        """Using the points in the clusters, determine the centroid
        (mean point) of each cluster"""
        members = [self.memberOf.count(i) for i in range(len(self.centroids))]
        self.centroids = [[sum([self.data[k][i]
                                for i in range(len(self.data[0]))
                                if self.memberOf[i] == centroid])/members[centroid]
                           for k in range(1, len(self.data))]
                          for centroid in range(len(self.centroids))] 
            
        
    
    def assignPointToCluster(self, i):
        """ assign point to cluster based on distance from centroids"""
        min = 999999
        clusterNum = -1
        for centroid in range(self.k):
            dist = self.euclideanDistance(i, centroid)
            if dist < min:
                min = dist
                clusterNum = centroid
        # here is where I will keep track of changing points
        if clusterNum != self.memberOf[i]:
            self.pointsChanged += 1
        # add square of distance to running sum of squared error
        self.sse += min**2
        return clusterNum

    def assignPointsToCluster(self):
        """ assign each data point to a cluster"""
        self.pointsChanged = 0
        self.sse = 0
        self.memberOf = [self.assignPointToCluster(i)
                         for i in range(len(self.data[1]))]
        

        
    def euclideanDistance(self, i, j):
        """ compute distance of point i from centroid j"""
        sumSquares = 0
        for k in range(1, self.cols):
            sumSquares += (self.data[k][i] - self.centroids[j][k-1])**2
        return math.sqrt(sumSquares)

    def kCluster(self):
        """the method that actually performs the clustering
        As you can see this method repeatedly
            updates the centroids by computing the mean point of each cluster
            re-assign the points to clusters based on these new centroids
        until the number of points that change cluster membership is less than 1%.
        """
        done = False
 
        while not done:
            self.iterationNumber += 1
            self.updateCentroids()
            self.assignPointsToCluster()
            #
            # we are done if fewer than 1% of the points change clusters
            #
            if float(self.pointsChanged) / len(self.memberOf) <  0.01:
                done = True
        print("Final SSE: %f" % self.sse)

    def showMembers(self):
        """Display the results"""
        for centroid in range(len(self.centroids)):
            print ("\n\nClass %i\n========" % centroid)
            for name in [self.data[0][i]  for i in range(len(self.data[0]))
                          if self.memberOf[i] == centroid]:
                print (name)
        
##
## RUN THE K-MEANS CLUSTERER ON THE DOG DATA USING K = 3
###
# change the path in the following to match where dogs.csv is on your machine
km = kClusterer("C:/doc/ch8/dogs.csv.txt", 3)
km.kCluster()
km.showMembers()
