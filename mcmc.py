import numpy as np
import matplotlib.pyplot as plt

def spin2binary(spin):
    return (spin+np.ones(spin.shape))/2

def binary2spin(binary):
    return (2*binary-np.ones(binary.shape))

class hamiltonian():
    
    def __init__(self,sizelist,maxsep=1,coupling=[0,0,-1]):
        self.sizelist = sizelist
        self.dimension = len(sizelist)
        self.maxsep = maxsep
        self.coupling = coupling
        assert len(self.coupling)== self.maxsep+2
        
    def neighbor(self,position,nnearest):
        pass
    
    def neighborenergy(self,pos,configuration, divided = 0):
        spinsum = [1]
        for nearest in range(0,self.maxsep+1):
            nb = self.neighbor(pos,nearest)
            nbtotal = 0
            nbtotal = sum([configuration[tuple(nbpos)] for nbpos in nb])
            spinsum.append(nbtotal)
        coupling = np.array(self.coupling)
        if divided == 1:
            coupling[0] *= 2
            coupling[1] *=2
        spinsum = np.array(spinsum)
        return np.inner(coupling,spinsum)*configuration[tuple(pos)]
    
    def totalenergy(self,configuration):
        te = 0
        for pos in np.ndindex(configuration.shape):
            te += self.neighborenergy(list(pos),configuration, divided = 1)
        return te/2
               
        
        
class Ising(hamiltonian):
    
    def __init__(self,sizelist,maxsep=1,coupling=[0,-1]):
        super().__init__(sizelist,maxsep,coupling)
        
    def neighbor(self,position,nnearest=1):  # the default one is square lattice with PBC: only NN couplings works!
        if nnearest > self.maxsep or nnearest <0:
            raise Exception('no such coupling term in Hamiltonians')
        nlist = []
        if nnearest == 0:
            return np.array([position])
        elif nnearest == 1:
            for i in range(self.dimension):
                le = position[:]
                ri = position[:]
                le[i] = (position[i]-nnearest)%self.sizelist[i]
                nlist.append(le)
                ri[i] = (position[i]+nnearest)%self.sizelist[i]
                nlist.append(ri)
        elif nnearest > 1:
            nlist = self.customized_neighbor(position, nnearest)
        return np.array(nlist)
    
    def energy2d(self, spin):
        te = 0
        for i in range(spin.shape[0]-1):
            te += np.dot(spin[i],spin[i+1])
        te += np.dot(spin[spin.shape[0]-1],spin[0])
        for i in range(spin.shape[1]-1):
            te += np.dot(spin[:,i],spin[:,i+1])
        te += np.dot(spin[:,spin.shape[1]-1],spin[:,0])   
        te *= self.coupling[2]
        te += np.sum(spin)*self.coupling[0]
        te += np.sum(spin**2)*self.coupling[1]
        return te
 

class configuration():
       
    def __init__(self,hamiltonianobj):
        self.system = hamiltonianobj
        self.config = np.ones(self.system.sizelist,dtype=np.int64)
        self.sites = np.product(self.system.sizelist)
        self.count = 0
        
    def feed(self,outconfig):
        self.config = outconfig[:]
        self.measurement([],2)
    
    def measurement(self,positions,updated):
        pass
        
    def randompos(self):
        position = []
        for j in range(self.system.dimension):
            position.append(np.random.randint(0, self.system.sizelist[j]))
        return position
  
    def mcmove(self, beta, autotime, method = 'local_update', tool = []):
        ''' This is to execute the monte carlo moves '''
        if method == 'local_update':
            for i in range(autotime): 
                updated = 0
                position = self.randompos()
                s =  self.config[tuple(position)]
                cost = -2*self.system.neighborenergy(position,self.config) # this is dangerous now, not suitable for soun square term of energy
                if self.system.coupling[1]!=0:
                    cost += 2*self.system.coupling[1]
                if cost < 0:
                    s = -s
                    updated = 1 
                elif np.random.rand() < np.exp(-cost*beta):
                    s = -s
                    updated = 1 
                self.config[tuple(position)] = s
                self.count =self.count+1
                self.measurement([position], updated)
        elif method == 'wolff_update':
            self.wolff_update(beta, autotime)
        elif method == 'rbm_update':
            self.rbm_update(beta,autotime,tool[0], tool[1], tool[2])
        elif method == 'customized_update':
            self.customized_update(beta , autotime, tool)
    
    def wolff_update(self, beta, autotime):
        for _ in range(autotime):
            p = 1-np.exp(2*self.system.coupling[2]*beta)
            position = self.randompos()
            cluster = [position]
            boundary_old = [position]
            while(len(boundary_old) != 0):
                boundary_new=[]
                for pos in boundary_old:
                    spin = self.config[tuple(pos)]
                    nbposs = self.system.neighbor(pos,nnearest=1).tolist()
                    for nbpos in nbposs:
                        if self.config[tuple(nbpos)] == spin and (nbpos not in cluster):
                            if np.random.rand()<p:
                                boundary_new.append(nbpos)
                                cluster.append(nbpos)    
                boundary_old = boundary_new
            for pos in cluster:
                self.config[tuple(pos)] = -spin
            self.count = self.count+1
            self.measurement(cluster, 1)
            
    def rbm_update(self, beta, autotime, rbmobj, nosteps, binary):
        if binary == False:
            for _ in range(autotime):
                self.config = binary2spin(rbmobj.Gibbsupdate(np.array([spin2binary(self.config)]), nosteps)[0].reshape(*self.system.sizelist))
                self.count = self.count+1
                self.measurement([], 3)
        elif binary == True: # this is very dangerous, be careful! The configuration states is 01 for this update scheme
            for _ in range(autotime):
                self.config = rbmobj.Gibbsupdate(np.array([self.config]), nosteps)[0].reshape(*self.system.sizelist)
                self.count = self.count+1
                self.measurement([], 3)
            
    
    def customized_update(self, beta, autotime, tool):
        pass ## don't forget the counter and the measurement in the last part of the definition, the reference is the wolff one
    
    def simulate(self,beta=1,num=9,autotime=1, pretime=1,method='local_update'):   
        ''' This module simulates the Ising model with visulize plot on configurations'''
        f = plt.figure(figsize=(15, 15), dpi=80);    
        self.mcmove(beta, pretime,method)
        for i in range(num):
            self.mcmove(beta, autotime, method)
            self.configPlot(f, self.config, i+1,  i+1);
        plt.show() 
                    
    def configPlot(self, f, config, i,  n_): # only available for 2D case
        X, Y = np.meshgrid(range(self.system.sizelist[0]),range(self.system.sizelist[1]))
        sp =  f.add_subplot(3, 3, n_ )  
        plt.setp(sp.get_yticklabels(), visible=False)
        plt.setp(sp.get_xticklabels(), visible=False)      
        plt.pcolormesh(X, Y, self.config,cmap='gray');
        plt.title('Time=%d'%i); plt.axis('tight')    


class observable():
    def inito(self,sites,config):
        pass
    def updato(self,positions,sites,config):
        pass

class mag(observable):
    def inito(self,sites,config):
        return 1/sites*np.sum(config)
    def updato(self,positions,sites,config):
        delta = 0
        for position in positions:
            delta = delta + 2* config[tuple(position)]/sites
        return delta

class mcmeasure(configuration):  ## this class is designed for measurement:API, object of some Hamiltonian, list of object of some observables
    def __init__(self,hamiltonianobj,observableobjs):
        super().__init__(hamiltonianobj)
        self.obs = observableobjs
        self.values = [[ob.inito(self.sites,self.config) for ob in self.obs]]
    def measurement(self,positions,updated):
        if updated == 0:
            self.values.append(self.values[-1])
        elif updated == 1:
            delta =[ob.updato(positions,self.sites,self.config) for ob in self.obs]
            newvalues = []
            for i in range(len(delta)):
                newvalues.append(self.values[-1][i]+delta[i])
            self.values.append(newvalues)
        elif updated == 2:
            self.values = [[ob.inito(self.sites,self.config) for ob in self.obs]]
        elif updated == 3:
            self.values.append([ob.inito(self.sites,self.config) for ob in self.obs])


class writeconf2file(configuration):
    def __init__(self,hamiltonianobj,presteps,sepsteps=1,filename='configuration.npy'):
        super().__init__(hamiltonianobj)
        self.presteps = presteps
        self.sepsteps = sepsteps
        self.path = filename
    def measurement(self, positions, updated):
        if self.count> self.presteps and (self.count % self.sepsteps ==0):
            with open(self.path, "ab") as f:
                np.save(f, self.config)
    def readconf(self):
        return readnpy(self.path)
        

def readnpy(filename):
    with open(filename,'rb') as f:
        a = []
        while(1):
            try:
                b=np.load(f)
                a.append(b)
            except BaseException:
                break
    return np.array(a)           
        
            
def autocorrelation(series): 
    stds = np.std(series)
    means = np.mean(series)
    fm=np.fft.rfft(series[:]-means)/stds
    fm2=np.abs(fm)**2
    cm=np.fft.irfft(fm2, len(series)) 
    cm_2= cm / cm[0] 
    upper = int(np.ceil(0.02*len(series)))
    return np.sum(cm_2[:upper])

def analysis(series,n=1, message=True):
    series = list(map(lambda x:x**n, series))
    tau = autocorrelation(series)
    dev = np.std(series)*np.sqrt((1+2*tau)/(len(series)-1))
    if message == True:
        print('the expectation value, standard deviation and autocorrelation time are correspondingly:')
    return [np.mean(series),dev,tau]

