'''
The basic module for Restricted Boltzmann machine(RBM) and its generalizations.
'''

import numpy as np
from scipy.special import expit


def chunks(l, n):
    '''
    Generator for iteration which divides list l into n batches randomly.
    
    :param l: the list to be divided
    :param n: the integer for the number of elements in one batch
    :yields: list of batch size
    
    Note the last batch may be smaller if 1%n != 0.
    '''
    np.random.shuffle(l)
    for i in range(0, len(l), n):
        yield l[i:i+n]
        
def reshapeinput(inputarray):
    '''
    Reshape the inputarray to 1D.
    
    :param inputarray: array with any shape
    :returns: array with 1D shape
    '''
    if len(inputarray.shape) == 1:
        return inputarray
    elif len(inputarray.shape) > 1:
        return inputarray.reshape(np.prod(inputarray.shape))


def sampleinput(arrays):
    '''
    Sample the arrays elementwise.
    
    :param arrays: the input array whose elements are between 0 to 1 as probabilities
    :returns: the array with the same shape as the input with all elements whose values are 1 or 0
    
    '''
    parrays = np.random.rand(*arrays.shape)
    return 0.5*np.sign(arrays-parrays)+0.5*np.ones(arrays.shape)


class RBM():
    '''
    RBM class
    
    :param visible: a list with the visble layer size, eg. [28,28] for MNIST data
    :param hidden: a list with the hidden layer size

    Some trainning details are inspired by : https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    '''
    
    def __init__(self, visible, hidden):
        self.novisible = np.prod(visible)
        self.visible = visible
        self.nohidden = np.prod(hidden)
        self.hidden = hidden
        self.weights = 0.06*np.random.rand(self.nohidden,self.novisible)-np.full((self.nohidden,self.novisible),0.03)
        self.biasonhidden = 0.02*np.random.rand(self.nohidden)-np.full((self.nohidden),0.01)
        self.biasonvisible = 0.02*np.random.rand(self.novisible)-np.full((self.novisible),0.01)
        self.string = ('RBM model: \nvisible layer size: %s \nhidden layer size:  %s'%(self.visible,self.hidden)) 
    
    def getbias(self):
        '''
        Get bias of the model in the shape of visible and hidden layer.
        
        :returns: list of two array, the first one is bias on visible layer while the second array 
                  is bias on hidden layer
        '''
        return np.array([self.biasonvisible.reshape(tuple(self.visible)),self.biasonhidden.reshape(tuple(self.hidden))])
    
    def getweights(self):
        '''
        Get weights of the model in the shape of visible and hidden layer.
        
        :returns: weights array, eg. the shape is (28,28,10,5) for RBM([28,28],[10,5]) 
        '''
        return self.weights.reshape(tuple(self.hidden+self.visible))
    
    def summary(self):
        '''
        Print the basic information on this RBM.
        '''
        print(self.string)
        
    def __repr__(self):
        return self.string
    
    __str__ = __repr__
    
    def probabilityonvisible(self,visibledatas):
        '''
        Get the conditional activation probability of hidden layer based on given configuration of visible layer.
        
        :param visibledatas: array of visible configuration arrays (both in 1d or in the shape of visible layers are ok)
        :returns: array of probability 1d arrays with the number of elements the same as hidden neurons 
        '''
        biasb = np.array([self.biasonhidden for _ in range(len(visibledatas))])
        visibledatas = visibledatas.reshape(visibledatas.shape[0],self.novisible)
        return expit(visibledatas@self.weights.T+biasb)
    
    def probabilityonhidden(self,hiddendatas):
        '''
        Get the conditional activation probability of visible layer based on given configuration of hidden layer.
        
        :param hiddendatas: array of hidden configuration arrays (both in 1d or in the shape of hidden layers are ok)
        :returns: array of probability 1d arrays with the number of elements the same as visible neurons 
        '''
        biasb = np.array([self.biasonvisible for _ in range(len(hiddendatas))])
        hiddendatas = hiddendatas.reshape(hiddendatas.shape[0],self.nohidden)
        return expit(hiddendatas@self.weights+biasb)
    
    def sampleonvisible(self,visibledatas):
        '''
        Get one sample configuration of hidden layer based on given configuration of visible layer.
        
        :param visibledatas: array of visible configuration arrays (both in 1d or in the shape of visible layers are ok)
        :returns: array of 1d configuration arrays with the number of elements the same as hidden neurons 
        '''
        probability = self.probabilityonvisible(visibledatas)
        return sampleinput(probability)

    def sampleonhidden(self,hiddendatas):
        '''
        Get one sample configuration of visible layer based on given configuration of vhidden layer.
        
        :param visibledatas: array of hidden configuration arrays (both in 1d or in the shape of hidden layers are ok)
        :returns: array of 1d configuration arrays with the number of elements the same as visible neurons 
        '''
        probability = self.probabilityonhidden(hiddendatas)
        return sampleinput(probability)
    
    def energy(self,visibledata,hiddendata):
        '''
        Calculate the energy of the model given configuration of both layers.
        
        :param visibledata: array of configuration of visible layer (both 1d and visible layer shape are ok)
        :param hiddendata: array of configuration of hidden layer (both 1d and hidden layer shape are ok)
        :returns: real value of the energy
        '''
        visibledata, hiddendata = reshapeinput(visibledata), reshapeinput(hiddendata)
        return -(hiddendata@self.weights@visibledata+self.biasonvisible@visibledata+self.biasonhidden@hiddendata)
    
    def freeenergy(self, visibledata):
        '''
        Caculate the free energy of the model given visible data.
        
        :param visibledata: array of configuration of visible layer (both 1d and visible layer shape are ok)
        :returns: real value of the free energy of the visible configuration
        '''
        visibledata = reshapeinput(visibledata)
        return -np.dot(self.biasonvisible,visibledata) - np.sum(np.log(np.ones(self.nohidden)+np.exp(self.weights@visibledata+self.biasonhidden)))
        
    
    def randomvisible(self, no = 1, aim = 'D'):
        '''
        Provide random samples whose shape consistent with the model.
        
        :param no: integer for the number of samples one want to generate
        :param aim: string, 'D' for configuration generation while 'P' for probability generation
        :returns: array of arrays of configuration or probability with the shape of visible layer
        '''
        if aim == 'D':
            return np.array([np.random.randint(2, size=tuple(self.visible)) for _ in range(no)])
        elif aim == 'P':
            return np.array([np.random.rand(*self.visible) for _ in range(no)])
        
    def Gibbsupdate(self, visibledatas, nosteps=1):
        '''
        Gibbs update for the model: start from visible layer
        
        :param visibledatas: array of configuration of visible layer (both 1d and visible layer shape are ok)
        :param nosteps: integer for the Gibbs update steps
        :returns: list of two arrays, the first is configuration of visible layer 
                  and the second is for hidden layer

        Note one step is v->h->v, so the hidden layer configurations is half step before visble ones
        '''
        for i in range(nosteps):
            hiddendatas = self.sampleonvisible(visibledatas)
            visibledatas = self.sampleonhidden(hiddendatas)
        return [visibledatas,hiddendatas]
    
    def cdk(self, visibledatas, nosteps = 1):
        '''
        Modified Gibbs update used for CD-k training.
        
        :param visibledatas: array of configuration of visible layer (both 1d and visible layer shape are ok)
        :param nosteps: integer for the Gibbs update steps or the k in CD-k
        :returns: list of two arrays, the first is configuration of visible layer 
                  and the second is for hidden layer

        Note the difference between cdk update and Gibbs update. In the last step, the visible data are given 
        by probability intead of states and then we use the probability to calculate probability of hidden 
        layer as data for hidden layer which is half step later compared to visibledata.
        '''
        if nosteps>1:
            hiddendatas = self.Gibbsupdate(visibledatas, nosteps-1)[1]
        elif nosteps == 1:
            hiddendatas = self.sampleonvisible(visibledatas)
        visibledatas = self.probabilityonhidden(hiddendatas)
        hiddendatas = self.probabilityonvisible(visibledatas)
        return [visibledatas,hiddendatas]
    
    def fit(self, visibledatas, testdatas, batch = 20, epoch = 50, learningrate = 0.05,
            regulation1 = 0, regulation2 = 0, cdkstep = 1, debuglog = True):
        '''
        Fit the RBM.
        
        :param visibledatas: the array of arrays of datas for training, eg. the shape [60000,28,28] for training
        :param testdatas: the array of arrays of datas for training, eg. the shape [60000,20,20] for training
        :param batch: integer for the size of batch for SGD
        :param epoch: integer for the numbers of epochs of training
        :param learningrate: real value for the update rate
        :param regulation1: L1 regularization term
        :param regulation2: L2 regularization term
        :param cdkstep: integer value of k in CD-k training
        :param debuglog: boolean, true for information print after each epoch
        '''
        noepoch = 0
        notestdatas = len(testdatas)
        trainingdatas = visibledatas.copy()

        for _ in range(epoch):

            datasinbatchs = chunks(trainingdatas, batch)

            for visibled in datasinbatchs:
 
                length = len(visibled)
                visibled = visibled.reshape(length, self.novisible)      
                hiddend = self.probabilityonvisible(visibled)
                
                positivev = np.mean(visibled,axis=0)
                positiveh = np.mean(hiddend,axis=0)
                positivew = np.transpose(hiddend)@visibled
                
                [visiblem, hiddenm] = self.cdk(visibled,nosteps=cdkstep)
                negativew = np.transpose(hiddenm)@visiblem
                negativev = np.mean(visiblem,axis=0)
                negativeh = np.mean(hiddenm,axis=0)

                self.weights += self.mask((learningrate/length)*(positivew-negativew))
                if regulation1 != 0:
                    self.weights += self.mask(-(learningrate*regulation1)*np.sign(self.weights))
                if regulation2 != 0:
                    self.weights += self.mask(-(learningrate*regulation2)*self.weights)
            
                self.biasonvisible += learningrate*(positivev-negativev)
                self.biasonhidden += learningrate*(positiveh-negativeh)

            noepoch += 1
            if debuglog == True:
                print('-------------------')
                print('epoch: %s finished'%noepoch)
                print('the reconstruction error: %s'%self.error(testdatas))
                dfe = np.mean([self.freeenergy(visibledatas[i]) for i in range(notestdatas)])
                dft = np.mean([self.freeenergy(testdatas[i]) for i in range(notestdatas)])
                dfo = np.mean([self.freeenergy(np.random.randint(2, size=tuple(self.visible))) for i in range(notestdatas)])
                print('the free energy difference between training and evaluation set: %s'%(dfe-dft)) 
                print('the free energy difference between training and standard set: %s'%(dfe-dfo))
                print('the magnitude of weights and the updates:')
                print(np.sum(np.abs(self.weights)),np.sum(np.abs(learningrate*(positivew-negativew))))

    def mask(self, updateweights):
        '''
        Mask the updateweights in some pattern.
        
        :param updateweights: array in the shape of weights
        :returns: array in the shape of weights after some processing
        '''
        return updateweights
                
    def error(self,visibledatas):
        '''
        Calculate the reconstruction error of specified visible data after one Gibbs update.
        
        :param visibledatas: array of configuration of visible layer (both 1d and visible layer shape are ok)
        '''
        return np.sum(np.abs(self.Gibbsupdate(visibledatas,nosteps=1)[0]-visibledatas.reshape(visibledatas.shape[0],self.novisible)))/visibledatas.size
                
                
class localRBM(RBM):
    '''
    RBM with locality, where only the weights within windows are nonzero for each hidden layer neuron.
    
    :param visible: a list with the visble layer size, eg. [28,28] for MNIST data
    :param window: a list for the size of window, eg. [2,2]
    :param stride: a list for the size of stride, eg. [2,2]
    '''
    
    def __init__(self, visible, window, stride):
        self.dimension = len(visible)
        for i in range(self.dimension):
            assert (visible[i]-window[i])%stride[i] == 0
        self.window = window
        self.stride = stride
        self.hidden = [int((visible[i]-window[i])/stride[i]+1) for i in range(self.dimension)]
        super().__init__(visible, self.hidden)
        self.maskmatrix = self.getmask()
        self.weights = self.mask(self.weights)
    
    def getmask(self):
        '''
        Get the mask matrix with zero elements in required vanishing weights links.
        
        :returns: the array of the shape of weights with zero and one
        '''
        maskm = np.zeros(tuple(self.hidden+self.visible))
        for hiddenpos in np.ndindex(*self.hidden):
            for windowpos in np.ndindex(*self.window):
                visiblepos = tuple(np.multiply(np.array(hiddenpos),np.array(self.stride))+np.array(windowpos))
                maskm[hiddenpos+visiblepos] = 1
        return maskm.reshape(self.nohidden,self.novisible)
        
    def mask(self, updateweights):
        return np.multiply(self.maskmatrix, updateweights) 