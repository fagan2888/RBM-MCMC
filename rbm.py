import numpy as np
from scipy.special import expit


def chunks(l, n):
    np.random.shuffle(l)
    for i in range(0, len(l), n):
        yield l[i:i+n]
        
def reshapeinput(inputarray):
    if len(inputarray.shape) == 1:
        return inputarray
    elif len(inputarray.shape) > 1:
        return inputarray.reshape(np.prod(inputarray.shape))


def sampleinput(arrays):
    parrays = np.random.rand(*arrays.shape)
    return 0.5*np.sign(arrays-parrays)+0.5*np.ones(arrays.shape)


class RBM():
    
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
        return np.array([self.biasonvisible.reshape(tuple(self.visible)),self.biasonhidden.reshape(tuple(self.hidden))])
    
    def getweights(self):
        return self.weights.reshape(tuple(self.hidden+self.visible))
    
    def summary(self):
        print(self.string)
        
    def __repr__(self):
        return self.string
    
    __str__ = __repr__
    
    def probabilityonvisible(self,visibledatas):
        biasb = np.array([self.biasonhidden for _ in range(len(visibledatas))])
        visibledatas = visibledatas.reshape(visibledatas.shape[0],self.novisible)
        return expit(visibledatas@self.weights.T+biasb)
    
    def probabilityonhidden(self,hiddendatas):
        biasb = np.array([self.biasonvisible for _ in range(len(hiddendatas))])
        hiddendatas = hiddendatas.reshape(hiddendatas.shape[0],self.nohidden)
        return expit(hiddendatas@self.weights+biasb)
    
    def sampleonvisible(self,visibledatas):
        probability = self.probabilityonvisible(visibledatas)
        return sampleinput(probability)

    def sampleonhidden(self,hiddendatas):
        probability = self.probabilityonhidden(hiddendatas)
        return sampleinput(probability)
    
    def energy(self,visibledata,hiddendata):
        visibledata, hiddendata = reshapeinput(visibledata), reshapeinput(hiddendata)
        return -(hiddendata@self.weights@visibledata+self.biasonvisible@visibledata+self.biasonhidden@hiddendata)
    
    def freeenergy(self, visibledata):
        visibledata = reshapeinput(visibledata)
        return -np.dot(self.biasonvisible,visibledata) - np.sum(np.log(np.ones(self.nohidden)+np.exp(self.weights@visibledata+self.biasonhidden)))
        
    
    def randomvisible(self, no = 1, aim = 'D'):
        if aim == 'D':
            return np.array([np.random.randint(2, size=tuple(self.visible)) for _ in range(no)])
        elif aim == 'P':
            return np.array([np.random.rand(*self.visible) for _ in range(no)])
        
    def Gibbsupdate(self, visibledatas, nosteps=1):
        for i in range(nosteps):
            hiddendatas = self.sampleonvisible(visibledatas)
            visibledatas = self.sampleonhidden(hiddendatas)
        return [visibledatas,hiddendatas]
    
    def cdk(self, visibledatas, nosteps = 1):
        if nosteps>1:
            hiddendatas = self.Gibbsupdate(visibledatas, nosteps-1)[1]
        elif nosteps == 1:
            hiddendatas = self.sampleonvisible(visibledatas)
        visibledatas = self.probabilityonhidden(hiddendatas)
        hiddendatas = self.probabilityonvisible(visibledatas)
        return [visibledatas,hiddendatas]
    
    def fit(self, visibledatas, testdatas, batch = 20, epoch = 50, learningrate = 0.05,regulation1 = 0, regulation2 = 0, cdkstep = 1, debuglog = True):
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
        return updateweights
                
    def error(self,visibledatas):
        return np.sum(np.abs(self.Gibbsupdate(visibledatas,nosteps=1)[0]-visibledatas.reshape(visibledatas.shape[0],self.novisible)))/visibledatas.size
                
                
class localRBM(RBM):
    
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
        maskm = np.zeros(tuple(self.hidden+self.visible))
        for hiddenpos in np.ndindex(*self.hidden):
            for windowpos in np.ndindex(*self.window):
                visiblepos = tuple(np.multiply(np.array(hiddenpos),np.array(self.stride))+np.array(windowpos))
                maskm[hiddenpos+visiblepos] = 1
        return maskm.reshape(self.nohidden,self.novisible)
        
    def mask(self, updateweights):
        return np.multiply(self.maskmatrix, updateweights) 