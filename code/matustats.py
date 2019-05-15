import numpy as np
from scipy import stats
from scipy.special import digamma,polygamma, ncfdtr,ncfdtri


__all__=['lognorm','gamma','weibull','exgaus']
def latinSquare(N=4):
    U=np.zeros((2**(N),N),dtype=int)
    for i in range(N):
        U[:,i]= np.mod(np.arange(U.shape[0])/2**(N-1-i),2)
    return U

def invdigamma(x):
    ''' computes inverse of digamma function
        >>> x=np.linspace(0,10,11)
        >>> plt.plot(x,invdigamma(digamma(x))) '''
    m=x>=-2.22
    y=m*(np.exp(x)+0.5)-(1-m)/(x-digamma(1))
    if type(y) is np.ndarray: y[np.isnan(y)]=1
    elif np.isnan(y): y=1
    L=digamma(y)-x
    while np.min(L)>1e-8:
        y=y-L/polygamma(1,y)
        L=digamma(y)-x
    return y
    
    
def ab2gg(a,b):
    ''' transform from a,b parametrization to parameters
        E[log(X)] and E[log(1-X)] with X~beta(a,b)
    '''
    temp=digamma(a+b)
    return digamma(a)-temp,digamma(b)-temp


def gg2ab(g1,g2):
    # init conditions Lau and Lau (1991)
    #a=np.exp(-3.929+10.523*np.exp(g2)-3.026*np.power(np.exp(g1),3)
    #         +1.757*np.exp(np.exp(g2)*np.sqrt(np.exp(g1))))
    #b=np.exp(-3.895+1.222*np.sqrt(g2)-6.9056*np.power(np.exp(g1),3))
    t1=np.exp(g1);t2=np.exp(g2)
    #if (t1+t2)>=1:print(g1,g2) 
    assert( (t1+t2)<1)
    a=0.5*(1+t1-t2)/(1-t1-t2);b=0.5*(1+t2-t1)/(1-t1-t2)
    ab=np.array([a,b])
    ggt=np.array([g1,g2])
    ggc=np.array(ab2gg(ab[0],ab[1]))
    crit=np.linalg.norm(ggt-ggc)
    while crit>1e-10:
        temp=polygamma(1,ab[0]+ab[1])
        J=np.array([[(polygamma(1,ab[0])-temp)*ab[0],-temp*ab[1]],
                    [-ab[0]*temp,ab[1]*(polygamma(1,ab[1])-temp)]])
        ab=np.exp(np.log(ab)-np.linalg.inv(J).dot(ggc-ggt))
        ggc=ab2gg(ab[0],ab[1])
        crit=np.linalg.norm(ggt-ggc)
    assert(ab[0]>=0 and ab[1] >=0)
    return ab[0],ab[1]

def logit(p): 
    res=np.log(p/(1-p))
    assert(not np.any(np.isnan(np.log(p/(1-p)))))
    return res
    
def invlogit(x): return 1/(1+np.exp(-x))


###########################################################
# Distributions
###########################################################

def OLRpmf(beta,c):
    ''' returns p(x) for x in {0,1,...,K}
        c must be strictly increasing series
    '''
    c=np.array(c)
    assert np.all(np.diff(c)>0) 
    p=[1]+list(invlogit(beta-c))+[0]
    return -np.diff(p)
    
def OLRrvs(beta, c,n,size=1):
    ''' generate data with expected response
        frequency from n subjects
    '''
    # c must be strictly increasing
    c=np.array(c)
    assert np.all(np.diff(c)>0) 
    
    p=[1]+list(invlogit(beta-c))+[0]
    p=-np.diff(p)
    if size<1: 
        proposal=np.int32(p*n)
        restsort=np.argsort((p*n)-proposal)[::-1]
        #print(p*n,proposal,restsort)
        for i in range(n-proposal.sum()):
            proposal[restsort[i]]+=1
        assert(proposal.sum()==n)
        out=[]
        for g in range(proposal.size):
                out.extend([g]*int(proposal[g])) 
        return np.array(out)
    else: 
        dat=np.random.multinomial(n,p,size)
        v=range(dat.shape[1])
        out=[]
        for r in range(dat.shape[0]):
            out.append([])
            for g in range(dat.shape[1]):
                out[-1].extend([v[g]]*int(dat[r,g])) 
        return np.array(out)
        

def lognorm(mu=1,sigma=1,phi=0):
    ''' Y ~ log(X)+phi
        X ~ Normal(mu,sigma)
        mu - mean of X
        sigma - standard deviation of X
    '''
    return stats.lognorm(sigma,loc=-phi,scale=np.exp(mu))
    

def gamma(mu=1,sigma=1,phi=0):
    ''' Gamma parametrized by mean mu and standard deviation sigma'''
    return stats.gamma(a=np.power(mu/sigma,2),scale=np.power(sigma,2)/mu,loc=-phi)

def weibull(scale=1,shape=1,loc=0):
    '''  pdf =shape/scale* (x/scale+loc)**(shape-1)
        * exp(-(x/scale+loc)**shape)
    '''
    return stats.weibull_min(shape,scale=scale,loc=-loc)


def exgaus(x,mu,sigma,lamda):
    ''' Exponentially modiefied gaussian
    	mu - gaus mean, sigma - gaus std, lambda - rate of expon
    '''
    l=lamda/2.
    return l*np.exp(l*(mu+l*sigma**2/2-x))*stats.norm.cdf((x-mu-sigma**2*l)/sigma)


#######################################################
# TESTS
#######################################################
def yuenTrimmed(x,y,tr=0.2,alpha=0.05):
    ''' x,y - ndarrays with sample data
        alpha - alpha level of the two sided hyp. test
        tr - 2*tr proportion of data will be discarded/trimmed
        
        >>> x=np.random.randn(104) +0.5
        >>> y=np.random.randn(49)
        >>> yuenTrimmed(x,y)
    ''' 
    dat=[np.sort(x),np.sort(y)];h=[];q=[];n=[];ss=[]
    for i in range(2):
        n+=[dat[i].size]
        h+=[n[i]-2*np.floor(tr*n[i])]
        g=int(np.floor(tr * n[i]))
        dat[i][:g]=dat[i][g]; dat[i][-g:]=dat[i][-g-1]
        ss+=[np.square(dat[i]-dat[i].mean()).sum()]
        q+=[ss[i]/(h[i]*(h[i]-1))]
    df=(q[0] + q[1])**2/((q[0]**2/(h[0]-1)) + (q[1]**2/(h[1]-1)))
    dif=stats.trim_mean(x,tr)-stats.trim_mean(y,tr)
    test=np.abs(dif/np.sqrt(q[0]+q[1]))
    return 2*(1-stats.t.cdf(test,df=df))
    
def anova2Bx2B(dat,alpha=0.05):
    '''
        dat - RxNx4 array, where N is the size of each group with
            columns FAL1FBL1, FAL1FBL2, FAL2FBL1, FAL2FBL2
        returns Rx3 array with p-values for each of R cases
            and for ME factor A, ME factor B, interaction AxB
    '''
    assert(dat.shape[2]==4)
    res=np.zeros((dat.shape[0],4))
    N=dat.shape[1]
    y=np.concatenate([dat[:,:,0],dat[:,:,1],dat[:,:,2],dat[:,:,3]],axis=1).T
    tot=np.square(y-np.atleast_2d(y.mean(0))).sum(0)
    x=np.ones((4*N,4))
    x1=np.zeros(4*N);x1[2*N:]=1
    x2=np.zeros(4*N);x2[N:2*N]=1;x2[3*N:]=1
    x[:,1]=x1*x2
    x[:,2]=x1*(1-x2)
    x[:,3]=(1-x1)*x2
    res[:,0]=np.linalg.lstsq(x,y,rcond=None)[1]
    x=np.ones((4*N,2))
    x[:,1]=x1
    res[:,1]=tot-np.linalg.lstsq(x,y,rcond=None)[1]
    x[:,1]=x2
    res[:,2]=tot-np.linalg.lstsq(x,y,rcond=None)[1]
    res[:,3]=tot-res[:,0]-res[:,1]-res[:,2]
    mss=res/np.float32(np.array([4*N-4,1,1,1]))
    F=mss[:,1:]/np.atleast_2d(mss[:,0]).T
    return 1-stats.f.cdf(F,1,4*N-4), mss



def scheirerRayHare2Bx2B(dat,alpha=0.05):
    '''
        dat - RxNx4 array, where N is the size of each group with
            columns FAL1FBL1, FAL1FBL2, FAL2FBL1, FAL2FBL2
        returns Rx3 array with p-values for each of R cases
            and for ME factor A, ME factor B, interaction AxB
            
        Test-code
        y=np.array(np.array([[15.23,14.32,14.77,15.12,14.05, 15.48,14.13,14.46,15.62,14.23], 
            [15.19, 14.67,14.48,15.34,14.22,16.66, 16.27, 16.35, 16.93, 15.05], 
            [16.98, 16.43,15.95,16.73,15.62,16.53, 16.26, 15.69, 16.97, 15.37], 
           [17.12, 16.65, 15.73, 17.77, 15.52, 16.15, 16.86, 15.18, 17.96, 15.26]]).T,ndmin=3)
        scheirerRayHare2Bx2B(y)
        anova2Bx2B(y)

        # check the above against this R code
        #delivery.df = data.frame(
        #  Service = c(rep("Carrier 1", 20), rep("Carrier 2", 20)),
        #  Destination = rep(c(rep("Office 1", 10),rep("Office 2", 10)),2),
        #  Time = c(15.23, 14.32, 14.77, 15.12, 14.05,
        #  15.48, 14.13, 14.46, 15.62, 14.23, 15.19, 14.67, 14.48, 15.34, 14.22,
        #  16.66, 16.27, 16.35, 16.93, 15.05, 16.98, 16.43, 15.95, 16.73, 15.62,
        #  16.53, 16.26, 15.69, 16.97, 15.37, 17.12, 16.65, 15.73, 17.77, 15.52,
        #  16.15, 16.86, 15.18, 17.96, 15.26)
        #)
        #delivery.mod1 = aov(Time ~ Destination*Service, data = delivery.df)
        #summary(delivery.mod1)
    '''
    assert(dat.shape[2]==4)
    N=dat.shape[1]
    for r in range(dat.shape[0]):
        dat[r,:,:]=np.reshape(stats.rankdata(dat[r,:,:]),[N,4])
    res=np.zeros((dat.shape[0],4))
    y=np.concatenate([dat[:,:,0],dat[:,:,1],dat[:,:,2],dat[:,:,3]],axis=1).T
    tot=np.square(y-np.atleast_2d(y.mean(0))).sum(0)
    x=np.ones((4*N,4))
    x1=np.zeros(4*N);x1[2*N:]=1
    x2=np.zeros(4*N);x2[N:2*N]=1;x2[3*N:]=1
    x[:,1]=x1*x2
    x[:,2]=x1*(1-x2)
    x[:,3]=(1-x1)*x2
    res[:,0]=np.linalg.lstsq(x,y,rcond=None)[1]
    x=np.ones((4*N,2))
    x[:,1]=x1
    res[:,1]=tot-np.linalg.lstsq(x,y,rcond=None)[1]
    x[:,1]=x2
    res[:,2]=tot-np.linalg.lstsq(x,y,rcond=None)[1]
    res[:,3]=tot-res[:,0]-res[:,1]-res[:,2]
    mtot=tot/(N*4)
    H=res[:,1:]/np.array(mtot,ndmin=2).T
    return 1-stats.chi2.cdf(H,1)
    
def pcaEIG(A,highdim=None):
    """ performs principal components analysis 
     (PCA) on the n-by-p data matrix A
     Rows of A correspond to observations, columns to features/attributes. 

    Returns :  
    coeff :
    is a p-by-p matrix, each column contains coefficients 
    for one principal component.
    score : 
    the principal component scores ie the representation 
    of A in the principal component space. Rows of SCORE 
    correspond to observations, columns to components.
    latent : 
    a vector containing the normalized eigenvalues (percent variance explained)
    of the covariance matrix of A.
    Reference: Bishop, C. (2006) PRML, Chap. 12.1
    """
    A=np.array(A)
    n=A.shape[0];m=A.shape[1]
    highdim = n<m
    assert n!=m
    M = (A-A.mean(1)[:,np.newaxis]) # mean-center data
    if highdim:
        [latent,coeff] = np.linalg.eigh(np.cov(M))
        coeff=M.T.dot(coeff)
        denom=np.sqrt((A.shape[1]-1)*latent[np.newaxis,:])
        coeff/=denom #make unit vector length
    else:
        [latent,coeff] = np.linalg.eigh(np.cov(M.T))
    score = M.dot(coeff)
    latent/=latent.sum()
    # sort the data
    indx=np.argsort(latent)[::-1]
    latent=latent[indx]
    coeff=coeff[:,indx]
    score=score[:,indx]
    assert np.allclose(np.linalg.norm(coeff,axis=0),1)
    return coeff,score,latent
