import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_some_samples(X, Y, testX, testY, name_prefix='', N=10, title=None):
    dim1 = int(np.sqrt(N))
    dim2 = int(np.ceil(float(N)/float(dim1)))
    while dim1 * dim2 != N:
        dim1 -= 1
        dim2 = N/dim1

    fig, AX = plt.subplots(dim2, dim1*2, figsize=(6,12))
    fig.suptitle(title)
    AX = np.reshape(AX, [dim1 * dim2 * 2])
    
    X = X[:N]
    Y = Y[:N]
    testX = testX[:N]
    testY = testY[:N]

    X = np.concatenate([X, testX])
    Y = np.concatenate([Y, testY])

    for i, ax in enumerate(AX):
        setStr = 'train' if i < N else 'test'
        label = np.argmax(Y[i], axis=-1)
        title = '{} ({})'.format(label, setStr)
        ax.set_title(title)
        ax.axis('off')
        ax.imshow(np.reshape(X[i], [100,100]), cmap='gray')

    fig.savefig('{}_plot_samples.png'.format(name_prefix))

if __name__ == '__main__':
    for s in range(2):
        no = s+1
        for d in range(3):
            X = np.load('Latest/curr_{}-{}X_sort.npy'.format(d, no), 'r')
            Y = np.load('Latest/curr_{}-{}Y_sort.npy'.format(d, no), 'r')
            testX = np.load('Latest/curr_{}-test{}X_sort.npy'.format(d, no), 'r')
            testY = np.load('Latest/curr_{}-test{}Y_sort.npy'.format(d, no), 'r')

            title = 'Difficulty {}, $N_{{objs}}$ {}'.format(d, no)
            plot_some_samples(X, Y, testX, testY, 'Latest/curr_{}-{}'.format(no, d), title=title)
