import numpy as np
import pystan
from matusplotlib import saveStanFit, loadStanFit
from scipy import stats
from matustats import OLRrvs
from scipy.stats import scoreatpercentile as sap

gg2abStancode='''
functions{
    real[] ab2gg(real a, real b){
        real gg[2];
        real temp;
        temp=digamma(a+b);
        gg[1]=digamma(a)-temp;
        gg[2]=digamma(b)-temp;
        return gg;
    }
    
    real[] gg2ab(real g1,real g2){
        real a; real b; real t[2]; real c;
        real gc[2]; vector[2] temp;matrix[2,2] J;
        t[1]=exp(g1); t[2]=exp(g2);
        a=0.5*(1+t[1]-t[2])/(1-t[1]-t[2]); 
        b=0.5*(1+t[2]-t[1])/(1-t[1]-t[2]);
        gc=ab2gg(a,b);
        c= sqrt((g1-gc[1])^2+(g2-gc[2])^2); 
        while (c >1e-10){
            t[1]=trigamma(a+b);
            J[1,1]=a*(trigamma(a)-t[1]);
            J[1,2]=-b*t[1];
            J[2,1]=-a*t[1];
            J[2,2]=b*(trigamma(b)-t[1]);
            temp[1]=gc[1]-g1;
            temp[2]=gc[2]-g2;
            temp=J\\temp;
            a=exp(log(a)-temp[1]);
            b=exp(log(b)-temp[2]);
            gc=ab2gg(a,b);
            c= sqrt((g1-gc[1])^2+(g2-gc[2])^2);
        }
        gc[1]=a;gc[2]=b;
        return gc;
    }
}
'''

def checkGg2abStancode():
    model=gg2abStancode+'''generated quantities{
        real a;
        real b;
        real gg[2];
        real ab[2];
        a= uniform_rng(0,100);
        b= uniform_rng(0,100);
        gg=ab2gg(a,b);
        ab=gg2ab(gg[1],gg[2]);
    }
    '''
    sm=pystan.StanModel(model_code=model)
    fit=sm.sampling(iter=100, chains=4,thin=2,warmup=50,n_jobs=4,
        algorithm="Fixed_param")
    w=fit.extract()
    assert np.allclose(w['ab'][:,0],w['a'])
    assert np.allclose(w['ab'][:,1],w['b'])
    
def schnallDataOLR():
    model='''
    data {
        int<lower=2> K;// nr of levels
        int<lower=0> N;// nr of subjects
        int<lower=1> M;// nr of items
        // y - response of subject n to item m
        int<lower=1,upper=K> y[N,M]; 
        // x - first column 0=original,1=replication
        // second column 0=control,1=treatment
        int x[N,2];
    }
    parameters {
        vector[M] beta[2];
        real d[2];
        ordered[K-1] c;
    }
    transformed parameters{
        vector[M] tbeta[2];
        for (m in 1:M){
        for (k in 1:2){
            if (m==1 && k==1) tbeta[k][m]=0.0; 
            // fix difficulty of first item to zero
            else tbeta[k][m]=beta[k][m];
    }}}
    model {
    for (k in 1:(K-1)) c[k]~ uniform(-100,100);
    for (m in 1:M){
        for (n in 1:N) y[n,m] ~ ordered_logistic(-tbeta[x[n,1]+1][m]
            +d[x[n,1]+1]*x[n,2], c); // RT
    }}
    '''
    smSchnallOLR=pystan.StanModel(model_name='schnallOLR',model_code=model)
    print('Finished Compilation, Fitting model')
    D=np.loadtxt('schnallstudy1.csv',delimiter=',')
    D[:,1]=1-D[:,1]
    dat = {'y':np.int32(D[:,2:8])+1,'K':10,'M':6,
        'x':np.int32(D[:,[0,1]]),'N':D.shape[0] }
    fitSchnall = smSchnallOLR.sampling(data=dat,iter=5000, chains=6,
        thin=2,warmup=500,n_jobs=6,seed=4,refresh=-1)
    #print(fitSchnall)
    saveStanFit(fitSchnall,'schnall')
    print('Finished fitting smSchnallOLR')
    
def parameterRecoveryData():
    # simulate data
    w=loadStanFit('schnall')
    offset=np.array(w['c'][:,0]/2+w['c'][:,-1]/2,ndmin=2).T
    scale=np.array(np.abs(w['c'][:,0]-w['c'][:,-1])/2,ndmin=2).T
    bp=np.linspace(-2.2,2.2,51)
    b=bp*np.median(scale)+np.median(offset)
    d=np.median(w['d'][:,0])
    c=np.median(w['c'],axis=0)
    S=[]
    for bb in b:
        S.append([])
        for dd in [0,d]:
            res=OLRrvs(bb+dd,c,100,size=0)
            if np.all(res==9): res[0]=8
            if np.all(res==0): res[0]=1
            temp=np.histogram(res,np.arange(11)-0.5)[0]
            #print(res)
            S[-1].append(temp) 
    np.save('S',S)
    print('Created Data for parameter recovery')
def parameterRecoveryOLR():
    S=np.load('S.npy')
    print('compiling smOLR')
    model='''
    data {
        int<lower=2> K;
        int<lower=0> y1[K];// response control
        int<lower=0> y2[K];// response treatment
    }
    parameters {
        real<lower=-100,upper=100> d;
        ordered[K-1] c;
    }
    model {
    for (k in 1:(K-1)) c[k]~ uniform(-20,20);
    for (k in 1:K){
        target+=y1[k]*ordered_logistic_lpmf(k|0.0,c);
        target+=y2[k]*ordered_logistic_lpmf(k|d,c);
    }}
    '''
    smOLR=pystan.StanModel(model_name='OLR',model_code=model)
    print('Finished Compilation, Fitting model')
    for k in range(len(S)):
        dat = {'y1':S[k][0],'y2':S[k][1],'K':S[k][0].size}
        fit = smOLR.sampling(data=dat,iter=4000, 
            chains=6,thin=5,warmup=2000,n_jobs=6,seed=4,refresh=-1)
        saveStanFit(fit,'schnSimOLR%02d'%k)
    print('Finished fitting smOLR')
def parameterRecoveryBB():
    S=np.load('S.npy')
    print('compiling smBB')
    model=gg2abStancode+'''
    data {
        int<lower=2> K;
        int<lower=0> y1[K];// response control
        int<lower=0>y2[K];// response treatment
    }
    parameters {
        real<lower=0,upper=20> ab1[2];
        real<lower=0,upper=20> ab2[2];
    }
    transformed parameters{
        real<upper=0> gg1[2];
        real<upper=0> gg2[2];
        gg1=ab2gg(ab1[1],ab1[2]);
        gg2=ab2gg(ab2[1],ab2[2]);
    }
    model {
    for (k in 1:K){
        target+=y1[k]*beta_binomial_lpmf(k|K,ab1[1],ab1[2]);
        target+=y2[k]*beta_binomial_lpmf(k|K,ab2[1],ab2[2]);
    }}
    '''
    smBB=pystan.StanModel(model_name='BB',model_code=model) 
    print('Finished Compilation, Fitting model')

    for k in range(len(S)):
        dat = {'y1':S[k][0],'y2':S[k][1],'K':S[k][0].size}
        fit = smBB.sampling(data=dat,iter=4000,control={'adapt_delta':1.0},
            chains=6,thin=5,warmup=2000,n_jobs=6,seed=5,refresh=-1)
        saveStanFit(fit,'schnSimBB%02d'%k)
    print('\nFinished fitting smBB')        
def plot(figname='stan.png',dpi=300):
    from matusplotlib import figure, subplot,plt
    figure(size=3,aspect=0.3)
    il=['dog','trolley','wallet','plane','resume',
        'kitten','mean score','median score']
    w=loadStanFit('schnall')
    S=np.load('S.npy')
    offset=np.array(w['c'][:,0]/2+w['c'][:,-1]/2,ndmin=2).T
    scale=np.array(np.abs(w['c'][:,0]-w['c'][:,-1])/2,ndmin=2).T
    bp=np.linspace(-2.2,2.2,51)
    b=bp*np.median(scale)+np.median(offset)
    cs=np.median(w['c'],axis=0)
    d=np.median(-w['d'],axis=0)
    #xlm=[cs[0]-0.2,cs[-1]+0.2]
    xlm=b[[0,-1]];cls=[]
    for i in range(cs.size): cls.append('$c_%d$'%i)
    tmp=np.median(-w['tbeta'],axis=0)
    
    for j in range(2):
        ax=subplot(1,2,1+j)
        plt.plot(xlm,[-d[0],-d[0]])
        plt.xlim(xlm)
        ax.set_xticks(cs);
        ax.set_xticklabels(cls)
        plt.plot(tmp[0,:],-0.7*np.ones(6),'xg')
        plt.plot(tmp[1,:],-0.9*np.ones(6),'xr')
        #for k in range(tmp.shape[1]):
        #    plt.plot(tmp[0:,k],[-0.09,-0.11],'k',alpha=0.2)
        ds=np.zeros((len(S),3))*np.nan
        for i in range(len(S)):
            if j:
                wi=loadStanFit('schnSimBB%02d'%i)
                ds[i,:]=sap(wi['gg1'][:,1]-wi['gg2'][:,1],[2.5,50,97.5])
            elif j==0:
                wi=loadStanFit('schnSimOLR%02d'%i)
                ds[i,:]=sap(wi['d'],[2.5,50,97.5])           
        plt.plot(b,ds[:,1],'k')
        temp=[list(b)+list(b)[::-1],
            list(ds[:,0])+list(ds[:,2])[::-1]]
        ax.add_patch(plt.Polygon(xy=np.array(temp).T,
            alpha=0.2,fc='k',ec='k'))
        plt.ylim([[-2,4],[-2,4]][j])
        plt.ylabel('$c_u^\Delta$')
        plt.xlabel(['$c_u=-c_l$','$c_u$'][j])
        plt.title(['OLRM','Beta-Binomial'][j])
        plt.grid(axis='x')
    plt.savefig(figname,bbox_inches='tight',dpi=dpi)


if __name__ =='__main__':
    #schnallDataOLR()
    #parameterRecoveryData()
    #parameterRecoveryOLR()
    parameterRecoveryBB()
    
