from plot_samples import plot_some_samples

from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

class MultiMnistDataGenerator(object):
    def __init__(self, X, Y, testX, testY):
        self.X = np.reshape(X, [-1,28,28,1])
        self.YoneHot = Y
        self.Y = np.argmax(Y, -1)
        self.testX = np.reshape(testX, [-1,28,28,1])
        self.testYoneHot = testY
        self.testY = np.argmax(testY, -1)
        
        self.patch_size = 28
        
    def __call__(self, 
            num_sets=6,
            labels=[0,1,2,3,4,5,6,7,8,9], 
            imshape=[100,100,1], 
            possible_area=[100, 100],
            a_0=125,
            m_0=0.5,
            multi=2, 
            N_classes=11,
            single_train_size=10000, 
            single_test_size=2000, 
            sort_by='y',
            basename='complete'):
        
        self.num_sets = num_sets
        self.labels = labels
        self.imshape = imshape
        self.possible_area = possible_area
        self.a_0 = a_0
        self.m_0 = m_0
        self.multi = multi
        self.train_size = single_train_size
        self.test_size = single_test_size
        self.N_classes = N_classes
        self.sort_by = sort_by
        self.basename = basename
        
        self.single_difficulties = np.arange(np.ceil(self.num_sets/self.multi))
        self.difficultyRange = int(np.ceil(self.num_sets / self.multi))
        for i in range(1, self.multi + 1):
            print 'creating set {}'.format(i)
            for difficulty in range(self.difficultyRange):
                print 'with difficulty {}'.format(difficulty)
                self.generateMultiMNISTDataset(i, difficulty)


    def generateMultiMNISTDataset(self, numObjs, difficulty):

        print 'numObjs:', numObjs

        self.YoneHot = np.zeros([self.Y.size, self.N_classes])
        self.YoneHot[np.arange(self.Y.size), self.Y] = 1

        self.testYoneHot = np.zeros([self.testY.size, self.N_classes])
        self.testYoneHot[np.arange(self.testY.size), self.testY] = 1
        
        train_shape = [self.train_size] + self.imshape
        test_shape = [self.test_size] + self.imshape
        
        train_set = np.zeros(train_shape)
        test_set = np.zeros(test_shape)
        print train_set.shape
        
        train_labels = np.zeros([self.train_size, numObjs, self.N_classes])
        test_labels = np.zeros([self.test_size, numObjs, self.N_classes])
        print train_labels.shape
        
        train_label_mask = np.array([True if y in self.labels else False for y in self.Y])
        X_small = self.X[train_label_mask]
        Y_small = self.YoneHot[train_label_mask]
        test_label_mask = np.array([True if y in self.labels else False for y in self.testY])
        testX_small = self.testX[test_label_mask]
        testY_small = self.testYoneHot[test_label_mask]
            
        print 'mask size', train_label_mask.shape
        
        train_dict = {
                'X': X_small,
                'Y': Y_small,
                'newX': train_set,
                'newY': train_labels,
                'size': self.train_size
                }
        
        test_dict = {
                'X': testX_small,
                'Y': testY_small,
                'newX': test_set,
                'newY': test_labels,
                'size': self.test_size
                }

        ps = self.patch_size
        for dset in [train_dict, test_dict]:
            # generate Index tuples for picking numbers from the datasets
            X = dset['X']
            Y = dset['Y']
            indeces = self.generateIndexTuples(X.shape[0], numObjs, dset['size'])
            for i, (image, index_tuple) in tqdm(enumerate(zip(dset['newX'], indeces))):
                if not isinstance(index_tuple, tuple):
                    index_tuple = tuple(index_tuple)
                location_tuple = self.generateDistinctLocations(numObjs, difficulty)
                for (x, y), index in zip(location_tuple, index_tuple):
                    image[x:x+ps, y:y+ps] = X[index]

                index_tuple = self.sortIndexTuple(location_tuple, index_tuple)
                dset['newY'][i] = [Y[j] for j in index_tuple]
        
        
        sort = 'sort' if self.sort_by is not None else 'not_sort'
        np.save('{}_{}-{}X_{}.npy'.format(self.basename, difficulty, numObjs, sort), train_set)
        np.save('{}_{}-{}Y_{}.npy'.format(self.basename, difficulty, numObjs, sort), train_labels)
        np.save('{}_{}-test{}X_{}.npy'.format(self.basename, difficulty, numObjs, sort), test_set)
        np.save('{}_{}-test{}Y_{}.npy'.format(self.basename, difficulty, numObjs, sort), test_labels)

        return train_set, train_labels, test_set, test_labels
    
    def generateIndexTuples(self, num_samples, size, numObjs):
        indeces = np.random.choice(num_samples, [numObjs, size])
        print 'indeces.shape:', indeces.shape
        
        return indeces
    
    def generateDistinctLocations(self, numObjs, difficulty):
        
        delta = (np.array(self.imshape[:2]) - np.array(self.possible_area)) / 2.

        # first round: random tuple
        max_x, max_y = np.array(self.possible_area) - self.patch_size + delta
        min_x, min_y = delta

        if numObjs == 1:
            # Location drawn from beta-distribution, that broadens with 
            # difficulty
            if difficulty == 0:
                # Place exactly at center
                x1 = (max_x + min_x) / 2 
                y1 = (max_y + min_y) / 2
            else:
                # Place Beta distributed
                a = self.alpha(difficulty)
                x1, y1 = np.random.beta(a=a, b=a, size=[2])
                # Scale to desired range
                x1 = x1 * (max_x - min_x) + min_x
                y1 = y1 * (max_y - min_y) + min_y
        
            locations = [[x1, y1]]
            return np.array(locations, dtype=int)

        elif numObjs > 1:
            # Initial location drawn from uniform distribution
            x1 = np.random.randint(min_x,max_x)
            y1 = np.random.randint(min_y,max_y)

            locations = [[x1, y1]]

        for i in range(numObjs - 1):
            m = self.steepnessFromDifficulty(difficulty)

            def line(x, x1, y1, m):
                offset = y1 - x1*m
    	        return m*x + offset

            def draw_y(y1, max_y=max_y):
                y2 = np.random.randint(min_y,max_y)
                while y2 <= y1+self.patch_size and y2 > y1-self.patch_size:
                    y2 = np.random.randint(min_x, max_y)
                return y2

    	    accept = False
    	    while not accept:
    	        y2 = draw_y(y1)
    	        x_stop1 = line(y2, y1, x1, -m)
    	        x_stop2 = line(y2, y1, x1, m)
    	        x_min, x_max = np.sort([x_stop1, x_stop2])

    	        x_min = np.ceil(x_min)
                x_min = np.max([0, x_min])
                x_max = np.floor(x_max)
                x_max = np.min([max_x, x_max])

    	        if x_min == x_max:
    	            continue
    	        x2 = np.random.randint(x_min, x_max)
    	        accept = True

            locations.append([x2, y2])
            
            if np.sqrt((x1-x2)**2 + (y1-y2)**2) < 28:
                print x1, y1
                print x2, y2
                print '\n'
            
        return np.array(locations, dtype=int)

    def sortIndexTuple(self, location_tuple, index_tuple):
        if self.sort_by is None:
            return index_tuple

        idx = 0 if self.sort_by == 'x' else 1
        l_slice = location_tuple[:,idx]

        indeces = l_slice.argsort()
        index_tuple = np.array(index_tuple)[indeces]

        return index_tuple

    def steepnessFromDifficulty(self, difficulty):
        return self.m_0 * 10**(2*difficulty)

    def alpha(self, difficulty):
        d_range = self.difficultyRange
        d = difficulty
        a = self.a_0 * ((1 + float(d_range - d)) / float(d_range))**d

        return a



if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    X, Y = mnist.train.images, mnist.train.labels
    X = np.reshape(X, [-1, 28, 28, 1])
    testX, testY = mnist.test.images, mnist.test.labels
    testX = np.reshape(testX, [-1, 28, 28, 1])
    
    MM = MultiMnistDataGenerator(X, Y, testX, testY)
    
    MM(labels=[0,1,2,3,4,5,6,7,8,9], 
        imshape=[100,100,1], 
        possible_area=[100, 100],
        a_0=13,
        multi=2, 
        N_classes=11,
        sort_by='y',
        basename='Latest/curr')
    
    for s in range(2):
        no = s+1
        for d in range(3):
            X = np.load('Latest/curr_{}-{}X_sort.npy'.format(d, no), 'r')
            Y = np.load('Latest/curr_{}-{}Y_sort.npy'.format(d, no), 'r')
            testX = np.load('Latest/curr_{}-test{}X_sort.npy'.format(d, no), 'r')
            testY = np.load('Latest/curr_{}-test{}Y_sort.npy'.format(d, no), 'r')
    
            title = 'Difficulty {}, $N_{{objs}}$ {}'.format(d, no)
            plot_some_samples(X, Y, testX, testY, 'Latest/curr_{}-{}'.format(no, d), title=title)
