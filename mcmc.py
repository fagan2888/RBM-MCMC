import numpy as np
import matplotlib.pyplot as plt



def spin2binary(spin):
    '''
    Convert 1,-1 configuration for an array to 1,0 configurations (-1 to 0 while 1 keep unchanged).
    
    :param spin: thearray of any shape with elements 1,-1
    :returns: an array with the same shape as input but with binary 1,0 elements
    '''
    return (spin+np.ones(spin.shape))/2

def binary2spin(binary):
    '''
    Convert 1,0 configuration for an array to 1,-1 configurations (0 to -1 while 1 keep unchanged).
    
    :param binary: the array of any shape weith elements 1,0
    :returns: an array with the same shape as input but with 1,-1 elements
    '''
    return (2*binary-np.ones(binary.shape))

class hamiltonian():
    '''
    Setup Hamiltonian for Markov chain Monte Carlo study, 
    the best practice is to inherit this class to customize Hamiltonian.
    The hamiltonian class now is only suitable for lattice model with Ising type two spin interactions,
    apart from the external field term where only one spin is involved.
        
    :param sizelist: a list whose elements give the size information of the system, 
                         eg. [3,4,5] describes the system with size 3*4*5 in three dimension
    :param maxsep: an integer signals the furthest coupling distance, 
                       eg. 1 is usually for only NN coupling is considered
    :param coupling: a list gives the coupling strength for different coupling terms in the Hamiltonian
    :raise: assert len(coupling) == maxsep+2, the two extra terms of couplings are for external field 
                and square of the single spin
    '''
    
    def __init__(self,sizelist,maxsep=1,coupling=[0,0,-1]):
        self.sizelist = sizelist
        self.dimension = len(sizelist)
        self.maxsep = maxsep
        self.coupling = coupling
        assert len(self.coupling)== self.maxsep+2
        
    def neighbor(self,position,nnearest):
        '''
        Customize this function in subclass, to make such model suitable for any dimension or lattice structure.
        
        :param position: the list gives the position on the lattice
        :param position: an interger for the nth nearest neighbor, 0 is for the position it self
        :returns: the list of positions, whose elements are nth NN sites positions refer to input position
        '''
        pass
    
    def neighborenergy(self,pos,configuration, divided = 0):
        '''
        Calculate the energy on specified spin is invovled with its neighbor sites.
        
        :param pos: a list for the position coordinate in the lattice for the specified spin
        :param configuration: the array with the system sizelist shape 
                              whose elements give the states of spin on corresponding lattice positions
        :param divided: 0 is for all the energy the specified spin is invovled 
                        while 1 is for the average energy for this spin, where the neighbor spin couplings are halved
        :returns: the real value of neighbor energy
        '''
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
        '''
        Calculate the total energy of the system defined by the Hamiltonian given the specific spin configuration.
        
        :param configuration: the array with the system sizelist shape 
                              whose elements give the states of spin on corresponding lattice positions
        :returns: the real value of total energy of the system
        
        Note this function is in general true but can be rather slow for specific model, you may prefer to 
        implement another fast totalenergy fuction in its subclass.
        '''
        te = 0
        for pos in np.ndindex(configuration.shape):
            te += self.neighborenergy(list(pos),configuration, divided = 1)
        return te/2
               
        
        
class Ising(hamiltonian):
    '''
    Subclass from hamitonian. It defines Ising model on any dimension on cube lattice where 
    only NN coupling is implemented. One can inherit Ising class by implementing further neighbor 
    terms in neighbor() to simply generalize Ising model to more coupling terms.
    Terms of :math:`coupling_0 S_i` or :math:`coupling_1 S_i^2` are kept from the parent class,
    which correspond to the first two coupling strength.
    '''
    
    def __init__(self,sizelist,maxsep=1,coupling=[0,-1]):
        super().__init__(sizelist,maxsep,coupling)
        
    def neighbor(self,position,nnearest=1):  
        '''
        Neighbor geometry is defined, which is cube or square lattice with periodic boundary condition.
        
        :param position: a list gives the lattice position of specified spin
        :param nnearest: an integer value tells which group of nerighbors we want,
                         only 0 and 1 is allowed in this class
        :returns: the list whose elements are list of positions for nth neighbor spins
        '''
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
        '''
        The fast reimplementation of totalenergy, one may also overwrites the original function with the same name.
        
        :param spin: the array with the system size shape whose elements give spin states
        :returns: the real value of the total energy in the system
        '''
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
    '''
    The class who store the configuration of the system and carryies out Monte Carlo updates.

    :param hamiltonianobj: one instance from hamiltonian class or its subclasses
    '''
       
    def __init__(self,hamiltonianobj):
        self.system = hamiltonianobj
        self.config = np.ones(self.system.sizelist,dtype=np.int64)
        self.sites = np.product(self.system.sizelist)
        self.count = 0
        
    def feed(self,outconfig):
        '''
        Refresh the configurations of the system from an specified configuration
        
        :param outconfig: the configuration array with system size shape
        '''
        self.config = outconfig[:]
        self.measurement([],2)
    
    def measurement(self,positions,updated):
        '''
        To be implemented by subclasses of configuration() as middlewares, which runs after each step 
        of Monte Carlo update.
        
        :param positions: the list whose elements are lists for lattice positions whose spin states 
                          have been updated in this step of MC updates
        :param updated: integers can be taken as 0 to 3.
                         0 is for an initialize;
                         1 is for no update on configurations
                         2 is for local update where the calculation should base on positions
                         3 is for update where the calculation is carried globally irrelevant of updated positions
        '''
        pass
        
    def randompos(self):
        '''
        Pick up one position randomly for spin update.
        
        :returns: the list for spin position
        '''
        position = []
        for j in range(self.system.dimension):
            position.append(np.random.randint(0, self.system.sizelist[j]))
        return position
  
    def mcmove(self, beta, autotime, method = 'local_update', tool = [0,0,0]):
        ''' 
        This is the main module to execute the Monte Carlo moves 
        
        :param beta: a positive real value for the inverse of temperature, :math:`\\frac{1}{k_B T}`
        :param autotime: an integer for MC steps to conduct
        :param method: some string indicates the update method. 'local_update' is the default one.
                       Other methods implemented in this class include 'wolff_update', which is only
                       suitable for typical Ising model with only NN coupling terms and 'rbm_update'
                       whose update proposal is generated by Gibbs updates of RBM machine.
        :param tool: a list which may contains extra information that is needed for certain update scheme,
                      In 'rbm_update' case, the tool list contains three terms. The RBM machine object, 
                      the number of steps for Gibbs update for each MC update and a boolen for binary states.
        
        One can implement customized update and gives the input 'customized_input', which works by default.
        '''
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
        '''
        Wolff algorithm for global Monte Carlo updates, which is only desinged for pure Ising model.
        
        :param beta: the real value of inverse temperature
        :param antotime: an integer gives the steps of MC updates
        '''
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
        '''
        Monte Carlo update approach by RBM.
        
        :param beta: the real value for inverse temperature
        :param autotime: the integer for steps of MC updates
        :param rbmobj: the RBM objects for the scheme
        :param nosteps: the integer for number of steps of Gibbs updates in the rbm for one MC update
        :param binary: boolean, False for 1,-1 state updates while True for 1,0 updates, the latter is dangerous to use.
        '''
        if binary == False:
            for _ in range(autotime):
                self.config = binary2spin(rbmobj.Gibbsupdate(np.array([spin2binary(self.config)]), nosteps)[0].reshape(*self.system.sizelist))
                self.count = self.count+1
                self.measurement([], 3)
        elif binary == True: 
            for _ in range(autotime):
                self.config = rbmobj.Gibbsupdate(np.array([self.config]), nosteps)[0].reshape(*self.system.sizelist)
                self.count = self.count+1
                self.measurement([], 3)
            
    
    def customized_update(self, beta, autotime, tool):
        '''
        The customized update scheme one may implment in the subclass.
        The following code should be put in the last part of function:
        self.count = self.count+1
        self.measurement([], 3)
        '''
        pass 
    
    def simulate(self,beta=1,num=9,autotime=1, pretime=1,method='local_update'):   
        ''' 
        Simulate the Ising model with visulize plot on configurations,
        but it is only helpful for 2D systems. There are at most nine figures of configuration evolutions.
        
        :param beta: real value for inverse temperature
        :param num: integer for number of figures one want to present, 9 is the most
        :param autotime: integer for the intermidiate steps between two figures
        :param pretime: integer for the steps between initial configuration and the second plot
        :param method: string for MC update method: 'local_update','wolff_update','rbm_update','customized_update'
        '''
        f = plt.figure(figsize=(15, 15), dpi=80);    
        self.mcmove(beta, pretime,method)
        for i in range(num):
            self.mcmove(beta, autotime, method)
            self.configPlot(f, self.config, i+1,  i+1);
        plt.show() 
                    
    def configPlot(self, f, config, i,  n_): 
        '''
        Assistant function for visualize MC updates with simulate() fum. 
        '''
        X, Y = np.meshgrid(range(self.system.sizelist[0]),range(self.system.sizelist[1]))
        sp =  f.add_subplot(3, 3, n_ )  
        plt.setp(sp.get_yticklabels(), visible=False)
        plt.setp(sp.get_xticklabels(), visible=False)      
        plt.pcolormesh(X, Y, self.config,cmap='gray');
        plt.title('Time=%d'%i); plt.axis('tight')    


class observable():
    '''
    Class to customized physical observables which can be measured during the MCMC process.
    The best practice is to inherit this class for specific observables. One need to implement inito()
    and updato() functions to make the class work.
    '''
    def inito(self,sites,config):
        '''
        Provide the value of observable given the global state configurations. Not implement in this class!
        
        :param sites: integer for the total number of lattice sites 
        :param config:  the array for state configurations
        :returns: real value for the value of observable
        '''
        pass
    def updato(self,positions,sites,config):
        '''
        Provide the delta value of observable given the list of updated positions lists and 
        the global configurations after update 
        
        :param positions: list of list of spin positions which updates this time
        :param sites: integer for the total number of lattice sites 
        :param config: the array for state configurations
        :returns: real value for the delta value of physical observable
        '''
        pass

class mag(observable):
    '''
    Subclass from observable for magnetic moment in th system.
    '''
    def inito(self,sites,config):
        return 1/sites*np.sum(config)
    def updato(self,positions,sites,config):
        delta = 0
        for position in positions:
            delta = delta + 2* config[tuple(position)]/sites
        return delta

class mcmeasure(configuration):
    '''
    This class is designed for measurement in Monte Carlo, which stores the observable value in each MC step.
    
    :param hamiltonianobj: one instance from hamiltonian class as the guide Hamiltonian for MC
    :param observableobjs: a list of instances from observable subclasses, eg. [mag()]
    '''
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
    '''
    Class for MC updates and export configurations to .npy file.

    :param hamiltonianobj: one instance from hamiltonian class as the guide Hamiltonian for MC
    :param presteps: integer for MC steps we drop before record
    :param sepsteps: integer for MC steps between two records
    :param filename: string for the file path, note delete the file if it already exists
    '''
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
        '''
        Read the .npy file written by this instance of class.
        '''
        return readnpy(self.path)
        

def readnpy(filename):
    '''
    Read multiple arrays from .npy files, and compose them as an array.
    
    :param filename: the string for file path
    :returns: the array with the shape (number of array, original array shape)
    '''
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
    '''
    Determine the auto correlation time from time series.
    
    :param series: an 1d array which can be viewed as time series
    :returns: the real value of autocorrelation time of the input array
    '''
    stds = np.std(series)
    means = np.mean(series)
    fm=np.fft.rfft(series[:]-means)/stds
    fm2=np.abs(fm)**2
    cm=np.fft.irfft(fm2, len(series)) 
    cm_2= cm / cm[0] 
    upper = int(np.ceil(0.02*len(series)))
    return np.sum(cm_2[:upper])

def analysis(series,n=1, message=True):
    '''
    The detailed analysis for time series including the estimatation on mean value, error bar
    and auto correlation time.
    
    :param series: an 1d array which can be viewed as time series
    :param n: a real value indicate we want to analyse the series as the power n of input series
    :param message: boolean value whose default value is true.
                    True for some information on the screeen indicating the return value,
                    False for turning off the print.
    :returns: the list contains three real values, 
              for expectation, error bar and autocorrelation time respectively.
    '''
    series = list(map(lambda x:x**n, series))
    tau = autocorrelation(series)
    dev = np.std(series)*np.sqrt((1+2*tau)/(len(series)-1))
    if message == True:
        print('the expectation value, standard deviation and autocorrelation time are correspondingly:')
    return [np.mean(series),dev,tau]

