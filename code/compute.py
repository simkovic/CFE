import numpy as np
import pylab as plt
from scipy import stats,__version__
from scipy.stats import scoreatpercentile as sap
from scipy.integrate import quad
from scipy.special import gamma,binom,beta 
from multiprocessing import Pool
import os,sys
from matustats import invdigamma,ab2gg,gg2ab,logit,invlogit, \
    OLRpmf,OLRrvs,yuenTrimmed,anova2Bx2B,scheirerRayHare2Bx2B

np.random.seed(6)
NREPS=10000 # number of repetitions
NPARTS=100 # total number of participants 
NCL=50 # number of c_l values 
DPATH='../data/' # data path
# research scenarios
S=[ [-1,0,np.nan,np.nan],
    [0,-1,-2,np.nan],
    [0,-1,-2,-3],
    [0,-2,-2,0],
    [0,-1,-2,-4],
    [-1,0,-2,-4],
    [-1,-4,-2,0],
    [0,-6,-12,-18],
    [0,-6,-12,-24],
    ] 
    
FIGDIR='figures'+os.path.sep
SUPDIR=FIGDIR+'supplement'+os.path.sep
CLRS=['b','r','g','y','k']  
DLBLS=['Gen. Gamma','Wald','Beta Prime','Beta','Beta-Binomial','OLRM']
DPI=300
FIGLBLS=['skew','cint','cohd','nhst','an1f','tost','bftt']

###################################################
#
# COMPUTATION
#
################################################### 
AI=27
def dgpWorker(np1,np2,cl,cld,rvs,tostDelta,R,N,ofn): 
    D=np.zeros([R,AI+9*(len(S)-2)])*np.nan          
    #print(np1,np2,cl)
    for r in range(R):
        dat=[]
        for g in range(2):
            try: dat.append(rvs(cl,np1*cld[0,g],np2,int(N/2)))
            except AssertionError: return np.zeros(D.shape)*np.nan
            D[r,[2*g,2*g+1]]=[np.mean(dat[-1][0]),np.var(dat[-1][0],ddof=1)]
            D[r,[2*g+4,2*g+5]]=[np.mean(dat[-1][1]),np.var(dat[-1][1],ddof=1)]
            D[r,22+g]=stats.skew(dat[-1][1])
        dif=dat[1][1]-dat[0][1]
        D[r,24]=np.var(dif)
        D[r,25]=stats.skew(dif)
        # check for invalid par combination, relevant for beta dist.
        if len(dat[0])==0 or len(dat[1])==0: continue
        for m in range(2):
            # t-test
            t,p=stats.ttest_ind_from_stats(D[r,2+m*4],
                    np.sqrt(D[r,3+m*4]),N/2, D[r,m*4],
                    np.sqrt(D[r,1+m*4]),N/2, equal_var=False)
            D[r,8+m]=(t>0)*p/2+(1-p/2)*(t<=0)
            # bayes factor in favor of no difference
            t,p=stats.ttest_ind_from_stats(D[r,2+m*4],
                    np.sqrt(D[r,3+m*4]),N/2, D[r,m*4],
                    np.sqrt(D[r,1+m*4]),N/2, equal_var=True)
            nn=N/4;v=N-2
            f=lambda g: (2*np.pi*(1+nn*g))**-0.5* \
                ((1+(t**2)/(1+nn*g)/v)**(-(v+1)/2)*g**-1.5*np.exp(-1/2/g))
            try: denom=quad(f,0,np.inf)[0]
            except Warning: denom=np.nan
            D[r,16+m]=(1+t**2/v)**(-(v+1)/2)/denom
        # mann-whitney u-test
        if np.all(dat[0][1]==dat[0][1][0]) and np.all(dat[1][1]==dat[0][1][0]):
            D[r,10]=1
        else: D[r,10]=stats.mannwhitneyu(dat[0][1],dat[1][1],alternative='less')[1]
        # yuen trimmed means
        D[r,11]=yuenTrimmed(dat[0][1],dat[1][1])
        #TOST
        for g in range(2):
            for gg in range(2):
                t,p=stats.ttest_ind_from_stats(D[r,g*4+2]+
                    [-1,1][gg]*tostDelta[g],np.sqrt(D[r,g*4+3]),N/2,
                    D[r,g*4],np.sqrt(D[r,g*4+1]),N/2, equal_var=False)
                D[r,12+gg+g*2]=[(t<=0)*p/2+(1-p/2)*(t>0),
                                    (t>0)*p/2+(1-p/2)*(t<=0)][gg]
        # three groups
        dat1=[];dat2=[]
        for g in range(3): 
            a,b=rvs(cl,np1*cld[1,g],np2,int(N/3))
            dat1.append(a);dat2.append(b)
        D[r,18]=stats.f_oneway(*dat1)[1]
        D[r,19]=stats.f_oneway(*dat2)[1]
        try: D[r,20]=stats.kruskal(*dat2)[1]
        except ValueError: D[r,20]=1
        #plevene=stats.levene(*dat2)[0]
        #D[r,21]=D[r,19-int(plevene<0.05)]
    # 2x2 factor design
    
    for h in range(len(S)-2):
        dat1=[];dat2=[]
        for g in range(4): 
            a,b=rvs(cl,np1*cld[2+h,g],np2,(int(N/4),R))
            dat1.append(a);dat2.append(b)
        D[:,AI+h*9:AI+3+h*9]=anova2Bx2B(np.array(dat1).T)[0]
        D[:,AI+3+h*9:AI+6+h*9]=anova2Bx2B(np.array(dat2).T)[0]
        D[:,AI+6+h*9:AI+9+h*9]=scheirerRayHare2Bx2B(np.array(dat2).T) 
    D[:,AI-1]=N
    np.save(ofn,D)
    #return D
    
    
        
###################################################
#
# PLOTTING ROUTINES
#
###################################################
from matusplotlib import * 

def plotCI(x,G,ylim=[-0.5,1],alpha=0.05,K=0):
    N=G[0,0,AI-1]/2
    for ii in range(2):
        #if K and ii==0:            
        #    d,sd=_difLogitBinomProp(G[:,:,6],G[:,:,4],N)
        #    #print(np.median(G[0,:,6]))
        #    d=np.median(d,axis=1);sd=np.median(sd,axis=1)
        #else:
        d=np.median(G[:,:,2+ii*4]-G[:,:,ii*4],axis=1)
        v1=G[:,:,3+ii*4];v2=G[:,:,1+ii*4]
        sd= np.sqrt(np.median(v1+v2,axis=1))/np.sqrt(2*N)
        df=np.floor(np.square((1+np.nanmedian(v1/v2,1))/N)/
                ((1+np.square(np.nanmedian(v1/v2,1)))/(N**2*(N-1))))
        sel=~np.isnan(d)
        C2=[CLRS[-1],CLRS[1]];C=[CLRS,C2]           
        plt.plot(x[sel],d[sel],C[K>0][ii],lw=[1.5,1,1][ii])
        xx=np.concatenate([x[sel],x[sel][::-1]])
        if K and ii==0:
            lb=d+sd*stats.norm.ppf(alpha/2)
            ub=d+sd*stats.norm.ppf(1-alpha/2)
        else:
            lb=d+sd*stats.t.ppf(alpha/2,df)
            ub=d+sd*stats.t.ppf(1-alpha/2,df)
        ci=np.concatenate([lb[sel],ub[sel][::-1]])
        if len(xx)==0: return
        plt.gca().add_patch(plt.Polygon(np.array([xx,ci]).T,
                alpha=0.2,fill=True,fc=C[K>0][ii],ec=C[K>0][ii]))
    plt.ylim(ylim) 
def plotCohendCI(x,G,ylim=[-0.3,1.5],alpha=0.05):
    N=G[0,0,AI-1]/2
    for ii in range(2):
        v1=G[:,:,3+ii*4];v2=G[:,:,1+ii*4]
        d=np.median((G[:,:,2+ii*4]-G[:,:,ii*4])/np.sqrt((v1+v2)/2),axis=1)
        sel=~np.isnan(d)
        plt.plot(x[sel],d[sel],CLRS[ii],lw=[1.5,1,1][ii])
        z=stats.norm.ppf(1-alpha/2)
        s=z*np.sqrt(2/N+np.square(d[sel])/4/N)+1e-15
        xx=np.concatenate([x[sel],x[sel][::-1]])
        ci=np.concatenate([d[sel]-s,(d[sel]+s)[::-1]])
        if len(xx)==0: return
        plt.gca().add_patch(plt.Polygon(np.array([xx,ci]).T,
                alpha=0.2,fill=True,fc=CLRS[ii],ec=CLRS[ii]))
    plt.ylim(ylim)
        
     
def plotNHST(x,G,showLegend=False,alpha=0.05):
    sel=np.all(~np.isnan(G[:,:,0]),axis=1)
    y=(G[sel,:,:]<alpha).mean(1)
    for ii in range(4):
        plt.plot(x[sel],y[:,ii],CLRS[ii],lw=[1.5,1,1,1][ii])
    #print(np.max(np.atleast_2d(y[:,2]).T-y,axis=0))
    if showLegend: plt.legend(['log welch','welch','MW test','trimmed t'],loc=2)
     
def plotTOST(x,G,showLegend=False,alpha=0.05):
    for i in range(2):
        sel=np.logical_or(np.all(~np.isnan(G[:,:,2*i+0]),axis=1),
            np.all(~np.isnan(G[:,:,2*i+1]),axis=1))    
        plt.plot(x[sel],np.logical_and(G[:,:,2*i]<alpha,
            G[:,:,2*i+1]<alpha).mean(1)[sel],CLRS[i],lw=[1.5,1][i])
    if showLegend: plt.legend(['log TOST','TOST'],loc=2)
def plotBFT(x,G,showLegend=False,alpha=0.05):
    plt.plot(x,np.median(G[:,:,0]/(1+G[:,:,0]),axis=1),CLRS[0],lw=1.5)
    plt.plot(x,np.median(G[:,:,1]/(1+G[:,:,1]),axis=1),CLRS[1],lw=1)
    if showLegend: plt.legend(['log Bayes t test','Bayes t test'],loc=3)
def plotANOVA1F(x,G,showLegend=False,alpha=0.05):
    sel=np.all(~np.isnan(G[:,:,0]),axis=1)
    y=(G[sel,:,:]<alpha).mean(1)
    for ii in range(3):
        plt.plot(x[sel],y[:,ii],CLRS[ii],lw=[1.5,1,1,1][ii]) 
    #print(np.max(np.atleast_2d(y[:,2]).T-y,axis=0))   
    if showLegend: plt.legend(['log-ANOVA','ANOVA','KW test'],loc=2)
def plotANOVAall(dgps,showLegend=False,alpha=0.05,i=2,j=2):        
    cls=['$c_l$','$b$','$c_l$','$c_l$','$c_l$','$c_l$']
    lblls=['no X','no ME','uncrossed','crossed',
        'double-crossed','no X','uncrossed']
    K=len(S)-4
    figure(size=3,aspect=1)
    for h in range(len(dgps)):
        D=dgps[h].loadData(i=i,j=j)[:,:,AI:]
        x=dgps[h].cl
        for k in range(K):
            ax=subplot(K,len(dgps),1+k*len(dgps)+h)
            plt.grid(axis='x')
           
            for ii in range(3):
                s=np.all(~np.isnan(D[:,:,k*9]),axis=1)
                if not k: plt.title(DLBLS[h])
                plt.plot(x[s],(D[s,:,k*9+ii*3+2]<alpha).mean(1),
                    CLRS[ii],lw=[3,1,1][ii],alpha=0.7)
                if k<5:
                    plt.plot(x[s],(D[s,:,k*9+ii*3+1]<alpha).mean(1),
                        CLRS[ii]+'--',alpha=0.7,lw=[3,1,1][ii])
                    plt.plot(x[s],(D[s,:,k*9+ii*3]<alpha).mean(1),
                        CLRS[ii]+':',alpha=0.7,lw=[3,1,1][ii])
            if not h: plt.ylabel(lblls[k])
            else:ax.set_yticklabels([])
            if k<K-1:ax.set_xticklabels([])
            else: plt.xlabel(cls[h])
            plt.ylim([-0.05,1.05])
            plt.xlim([x[0],x[-1]])
            plt.locator_params(axis='x', nbins=4)#,min_n_ticks=3)
    plt.savefig(FIGDIR+'ANOVA%d%d.png'%(i,j),bbox_inches='tight',dpi=DPI)

def plotBrief(x,R,y=np.nan,pref='', ylim=[[],[]]):        
    figure(size=2,aspect=1.2)
    for f in range(1,7):  
        ax=subplot(3,2,f);
        plt.grid(axis='x')
        plt.ylim([-0.02,1.02])
        plt.title(['CI','Cohen\'s $d$','TOST','Bayesian $t$ test','Two-group tests','Three-group tests'][f-1])
        if f==1: 
            plotCI(x,R)
            #if np.any(~np.isnan(y)): plt.plot(x,y,'c')
            plt.ylim(ylim[0])
        elif f==2: 
            plotCohendCI(x,R)
            plt.ylim(ylim[1])
        elif f==3: plotTOST(x,R[:,:,12:16])
        elif f==4: plotBFT(x,R[:,:,16:18])
        elif f==5: 
            plotNHST(x,R[:,:,8:12])
        elif f==6: plotANOVA1F(x,R[:,:,18:21])
        if f<5: ax.set_xticklabels([])
        else: plt.xlabel(['$c_l$','$b$'][int(pref=='W')])
        plt.xlim([x[0],x[-1]])
    plt.savefig(FIGDIR+pref+'brief',dpi=DPI,bbox_inches='tight')
       
def plotXtypes(S):
    figure(size=2,aspect=0.4)
    lbls=['No X','Ballanced','Uncrossed','Crossed','Double-crossed']
    for i in range(5):
        s=S[i+2]
        ax=subplot(1,5,i+1)
        plt.plot([0,1],s[:2],'-o')
        plt.plot([0,1],s[2:],'-o')
        ax.set_xticks([0,1])
        ax.set_xticklabels(['G1','G2'])
        plt.ylim([-4.5,0.5])
        plt.xlim([-0.25,1.25])
        if i>0: ax.set_yticklabels([])
        else: 
            #plt.legend(['F1','F2'])
            plt.ylabel('$d$')
        plt.xlabel(lbls[i])
    plt.savefig(FIGDIR+'xtypes.png',bbox_inches='tight',dpi=DPI)
    plt.clf()
    plt.close()

def plotPDF(A,B,C,pdf,xmax=1,albls=['','','test']):
    C=np.linspace(C[0],C[-1],6)
    x=np.linspace(0,xmax,101)
    figure(size=3,aspect=1)
    def _hlp(A,i):
        if A.size>1 and xmax>0: return '%.2f'%A[i]
        else: return ''
    for i in range(A.size):
        for j in range(B.size):
            for h in range(C.size):
                ax=subplot(A.size,B.size,B.size*i+1+j)
                try:
                    if xmax>0: y=pdf(x,A[i],B[j],C[h])
                    else: y,x=pdf(A[i],B[j],C[h])
                except AssertionError: continue
                plt.plot(x,y,'k',alpha=0.2)
                plt.ylim([0,1])
                ax.set_yticks([0,1])
                if j>0: ax.set_yticklabels([])
                else: plt.ylabel(albls[0]+_hlp(A,i))
                if i+1<A.size: ax.set_xticklabels([])
                else: plt.xlabel(albls[1]+_hlp(B,j))
                plt.grid(False)
    plt.savefig(SUPDIR+albls[-1]+'prob.png',      
        bbox_inches='tight',dpi=DPI)
    plt.clf();plt.close()
def plotFcfe():
    figure(size=1,aspect=0.6)
    x=np.linspace(0,1,101)[1:-1]
    fl=[-np.log(x),-np.log(x/(1-x))]
    fu=[-np.log(1-x),-np.log((1-x)/x)]
    for i in range(2):
        ax=subplot(1,2,1+i)
        plt.plot(x,fl[i])
        plt.plot(x,fu[i])
        plt.xlabel('$\phi$')
        #if not i:plt.ylabel('$f(\phi)$')
        plt.text(0.25,3,'$f_l(\phi)$',horizontalalignment='center')
        plt.text(0.65,4.1,'$f_u(1-\phi)$',horizontalalignment='center')
        ttl=['$-\log x$','$-\log(x/(1-x))$'][i]
        #plt.legend(['$f_l(\phi)$','$f_u(1-\phi)$'])
        plt.title(ttl,fontsize=10) 
    plt.savefig(FIGDIR+'fcfe.png',bbox_inches='tight',dpi=DPI)
    plt.clf();plt.close()
    
def plotSkew(dgps):
    cls=['$c_l$','$b$','$c_l$','$c_l$','$c_l$','$c_l$']
    figure(size=3,aspect=0.6)
    for h in range(len(dgps)):
        try:D=dgps[h].loadData(i=2,j=2)
        except FileNotFoundError: continue
        x=dgps[h].cl
        for g in range(3):
            ax=subplot(3,len(dgps),1+h+g*len(dgps))
            plt.grid(axis='x')
            if g==0: 
                plt.title(DLBLS[h])
                y=np.median(D[:,:,6],axis=1)*[1,0.1][dgps[h].suf=='BE']
            elif g==1: y=np.median(D[:,:,7],axis=1)
            elif g==2: y=np.median(D[:,:,23],axis=1)
            plt.plot(x,y,CLRS[-1],lw=1)
            if g==0: plt.ylim([0,1])
            elif g==1: plt.ylim([[0,10],[0,0.01],[0,3],[0,6],[0,0.1],[0,0.06]][h])
            elif g==2: plt.ylim([[0,3],[0,3],[0,8],[0,3],[0,3],[-3,3]][h])
            if g<2: ax.set_xticklabels([])
            else: plt.xlabel(cls[h])
            plt.xlim([x[0],x[-1]])
            ax.locator_params(axis='x', nbins=4,min_n_ticks=3)
            if not h: plt.ylabel(['Mean','Variance','Skewness'][g])
            elif g==0: ax.set_yticklabels([])
            elif g==1:
                lm=[0,-3,0,0,-2,-2][h]
                if lm!=0: ax.ticklabel_format(axis='y',style='sci',scilimits=(lm,lm))
    plt.savefig(FIGDIR+'skew.png',bbox_inches='tight',dpi=DPI)
    plt.clf();plt.close()
###################################################
#
# MAIN
#
###################################################


    
class DataGeneratingProcess():
    def __init__(self,cl,nuispar1,nuispar2,cldeltaScale,pdf,
        rvs,tost,lbls,cilim,pdfxmax):
        self.cl=cl
        self.np1=nuispar1
        self.np2=nuispar2
        self.cld=cldeltaScale*np.array(S)
        self.pdf=pdf
        self.rvs=rvs
        self.tost=tost
        self.suf=lbls[0]
        self.np2lbl=lbls[1]
        self.cil=np.array(cilim)
        self.xmax=pdfxmax
    def compute(self):
        pool=Pool(8)
        res=[]
        for i in range(self.np1.size):
            res.append([])
            for j in range(self.np2.size):
                res[-1].append([])
                ind=int(self.tost*self.cl.size) 
                d1=self.rvs(self.cl[ind],self.cld[0,1]*self.np1[i],
                        self.np2[j],5000)
                d0=self.rvs(self.cl[ind],self.cld[0,0]*self.np1[i],
                        self.np2[j],5000)
                tostDelta=[np.nan,np.nan]
                for k in range(2):
                    if not np.all(np.isnan(d1[k]-d0[k])):
                        tostDelta[k]=np.median(d1[k]-d0[k])
                for s in range(self.cl.size):
                    ofn=DPATH+'D_%s%02d%02d%02d'%(self.suf,i,j,s)
                    temp=pool.apply_async(dgpWorker,
                        [self.np1[i],self.np2[j],self.cl[s],self.cld,
                        self.rvs,tostDelta,NREPS,NPARTS,ofn])
                    res[-1][-1].append(temp)
        pool.close()
        from time import sleep,time
        from matusplotlib import printProgress
        tot=self.np1.size*self.np2.size*self.cl.size
        t0=time();prevdone=-1
        while True:
            done=0
            for i in range(self.np1.size):
                for j in range(self.np2.size):
                    for s in range(self.cl.size):
                        done+=int(res[i][j][s].ready())
            if done>prevdone:
                printProgress(done,tot,time()-t0,self.suf+' running simulation\t')
                prevdone=done
            sleep(1)
            if done==tot: break
        np.save(DPATH+'cl_'+self.suf,self.cl)   
        #
        #for i in range(self.np1.size):
        #    for j in range(self.np2.size):
        #        for s in range(self.cl.size):
        #            ofn=DPATH+'D_%s%02d%02d%02d'%(self.suf,i,j,s)
        #            temp=res[i][j][s].get()
        #            np.save(ofn,temp)
        #np.save(DPATH+'D_'+self.suf,D)
    def loadData(self,i=2,j=2):
        R=np.zeros([self.cl.size,NREPS,AI+9*(len(S)-2)],dtype=np.float64)
        for s in range(self.cl.size):
            ifn=DPATH+'D_%s%02d%02d%02d'%(self.suf,i,j,s)
            R[s,:,:]=np.load(ifn+'.npy')
        return R
    def plot(self):
        print(self.suf+'creating figures')
        i=int(self.np1.size/2)
        j=int(self.np2.size/2);
        R=np.squeeze(self.loadData(i=i,j=j))
        tref=(self.cld[0,1]-self.cld[0,0])*self.np1[2]
        plotBrief(self.cl,R,y=tref,pref=self.suf,
            ylim=self.cil[[2,7]])
    def makeSupplement(self):
        print(self.suf+'creating supplement')
        #plotPDF(self.np1,self.np2,self.cl,self.pdf,xmax=self.xmax,
        #    albls=['$c_l^\Delta=$','$%s=$'%self.np2lbl,self.suf])
        #tref=(self.cld[0,1]-self.cld[0,0])*np.ones(self.cl.shape)
        n1=self.np1.size;n2=self.np2.size
        for f in range(1,7):
            plt.close('all')
            figure(num=f+1,size=3,aspect=1)
            for i in range(n1):
                for j in range(n2):
                    R=self.loadData(i=i,j=j)
                    ax=subplot(n1,n2,i*n2+j+1)
                    #plt.title(getTitle(i,j))
                    plt.grid(axis='x')
                    plt.ylim([-0.05,1.05])
                    x=np.load(DPATH+'cl_'+self.suf+'.npy')
                    legon= (i==0 and j==0)
                    if f==1:
                        plotCI(x,R)
                        tref=(self.cld[0,1]-self.cld[0,0])*self.np1[i]
                        plt.plot(x,tref*np.ones(x.size),'c')
                        plt.ylim(self.cil[i])
                    elif f==2:  
                        plotCohendCI(x,R,ylim=self.cil[i+5])
                    elif f==3:plotNHST(x,R[:,:,8:12],legon)
                    elif f==4:plotANOVA1F(x,R[:,:,18:22],legon)
                    elif f==5:plotTOST(x,R[:,:,12:16],legon)
                    elif f==6:plotBFT(x,R[:,:,16:18],legon)
                    plt.xlim([x[0],x[-1]])
                    if j!=0:ax.set_yticklabels([])
                    else:plt.ylabel('$c_l^\Delta=%.2f$'%self.np1[i])
                    if i!=n1-1: ax.set_xticklabels([])
                    else:
                        temp=(self.np2lbl,self.np2[j])
                        plt.xlabel('$%s=%.2f$'%temp)
            plt.savefig(SUPDIR+self.suf+FIGLBLS[f],
                dpi=DPI,bbox_inches='tight')
            plt.clf();plt.close()
            
        
#############################################       
# generalized gamma 
def GGpdf(x,np1,np2,cl): 
    return stats.gengamma.pdf(x,invdigamma(cl*np2),np2,0,1)
def GGrvs(cl,cld,cn,n):
    k=invdigamma(cl+cld*cn)
    res=stats.gengamma.rvs(k,cn,0,1,size=n)
    res[res==0]=np.exp(-20)
    return np.log(res),res  
cilims=[[-1,2],[-1,2],[-1,2],[-1,2.5],[-1,2.5],
    [-0.5,1.5],[-0.5,1.5],[-0.5,1.5],[-0.5,1.5],[-0.5,1.5]]    
gengamma=DataGeneratingProcess(np.linspace(-4,4,NCL),
    np.array([0.0625,0.125,0.25,0.5,1]),np.exp(np.linspace(-1,1,5)),
    1,GGpdf,GGrvs,0.75,['GG','c_n'],cilims,10)
#############################################
# wald 
def Wpdf(x,np1,np2,b,a=1):
    sig=np.square(a/np2)
    mm=a/b/sig
    return stats.invgauss.pdf(x,mm,loc=0,scale=sig)   
def Wrvs(b,bd,s,n,a=1):
    sig=np.square(a/s)
    mm=a/(b+bd)/sig
    res=stats.invgauss.rvs(mm,loc=0,scale=sig,size=n)
    res[res==0]=np.exp(-20)
    return np.log(res),res
cilims=[[-0.05,0.25],[-0.05,0.25],[-0.05,0.25],[-0.05,0.35],[-0.05,0.35],[-0.3,1.2],[-0.3,1.2],[-0.3,1.2],[-0.3,2],[-0.3,3]]
wald=DataGeneratingProcess(np.linspace(20.1,0.1,NCL),
    np.array([0.25,0.5,1,2,4]),np.exp(np.linspace(-1,1,5)),
    -1,Wpdf,Wrvs,0.75,['W','\sigma'],cilims,5)
   
#########################################################
#beta prime distribution

def BPpdf(x,np1,np2,cl):
    ga=cl-np2;gb=-np2
    a,b=gg2ab(ga,gb)
    #except AssertionError:
    #    return np.zeros(x.size)*np.nan
    return stats.betaprime.pdf(x,a,b,scale=1)
    
def BPrvs(cl,cld,cm,n):
    ga=cl+cld-cm;gb=-cm
    try: a,b=gg2ab(ga,gb)
    except AssertionError:return np.ones(n)*np.nan,np.ones(n)*np.nan
    res=stats.betaprime.rvs(a,b,scale=1,size=n)
    res[res==0]=np.exp(-20)
    return np.log(res),res
cilims=[[-1.5,3.5],[-1.5,3.5],[-1.5,3.5],[-3,7],[-4,10],[-0.6,1],[-0.6,1],[-0.6,1],[-0.6,1],[-0.6,1]]   
betaprime=DataGeneratingProcess(np.linspace(-10,-2.5,NCL),
    np.array([0.5,1,2,4,8]),np.linspace(0.01,0.3,5),
    1,BPpdf,BPrvs,0.75,['BP','c_m'],cilims,1)
      
#########################################################
#beta distribution

def BEpdf(x,np1,np2,cl):
    try: a,b=gg2ab(cl,np2)
    except AssertionError: return np.nan*np.zeros(x.size)
    return stats.beta.pdf(x,a,b,0,scale=1)

def BErvs(cl,cld,cu,n):
    try: a,b=gg2ab(cl+cld,cu)
    except AssertionError: return np.ones(n)*np.nan,np.ones(n)*np.nan 
    res=stats.beta.rvs(a,b,0,scale=10,size=n)
    res[res==10]=10-np.exp(-20)
    res[res==0]=np.exp(-20)
    res0=logit(res/10.)
    return res0,res
cilims=[[-1,4],[-1,4],[-1,4],[-2,6],[-3,8],[-0.5,1],[-0.5,1],[-0.5,1],[-0.5,1],[-0.5,1]]
betadgp=DataGeneratingProcess(np.linspace(-20,-0.7,NCL),
    np.array([1,2,3,4,5]),np.array([-0.7,-0.3,-0.2,-0.15,-0.1]),
    1,BEpdf,BErvs,0.75,['BE','c_u'],cilims,1)

###################################################### 
#binomial-beta distribution with geometric pars
def BBpmf(np1,np2,cl,gb=-0.3):
    n=np2
    try: a,b=gg2ab(cl,gb)
    except AssertionError: 
        return np.nan*np.zeros(n),np.arange(n)
    res=[]
    for k in range(n):
        res.append(binom(n-1,k)*beta(k+a,n-k-1+b)/beta(a,b))
    return res,np.arange(n)
        
def BBrvs(cl,cld,cn,n,cu=-0.3):
    if type(n) is int: 
        rr=1;nn=n
    else: 
        rr=n[1];nn=n[0]
    a,b=gg2ab(cl+cld,cu)
    res=stats.beta.rvs(a,b,0,1,size=(nn,rr))
    res[res==1]=1-np.exp(-20)
    res[res==0]=np.exp(-20)
    res=np.random.binomial(cn-1,res)
    res0=(2*res+1)/2./float(cn)
    res0=logit(res0)
    return res0,res/(cn-1) 
cilims=[[-0.5,0.7],[-0.5,0.7],[-0.5,0.7],[-0.25,1],[-0.25,1],[-0.5,1],[-0.5,1],[-0.5,1],[-0.5,1],[-0.5,1]]
betabinom=DataGeneratingProcess(np.linspace(-5.5,-1.5,NCL),
    np.array([1,2,4,6,8]),np.array([3,5,7,9,15]),
    1,BBpmf,BBrvs,0.75,['BB','n'],cilims,-1)   
########################################
#ordinal logistic regression model
def OLpmf(np1,np2,cl):
    c=np.linspace(-3,3,np2-1)
    return OLRpmf(cl,c),np.arange(c.size+1)
def OLrvs(cl,cld,cn,n):
    if type(n) is int: 
        rr=1;nn=n
    else: 
        rr=n[1];nn=n[0]
    c=np.linspace(-3,3,cn-1) 
    res=OLRrvs(cl+cld,c,nn,size=rr)
    res=np.squeeze(res)
    res0=(2*res+1)/2./float(cn)
    res0=logit(res0)
    return res0.T,res.T/float(cn-1)   
cilims=[[-0.1,0.3],[-0.1,0.3],[-0.1,0.5],[-0.1,0.6],[-0.1,0.6],[-0.3,1.2],[-0.3,1.2],[-0.3,1.2],[-0.3,1.2],[-0.3,1.2]]
olrm=DataGeneratingProcess(np.linspace(-5,5,NCL),
    np.array([0.4,0.6,0.8,1,1.2]),np.array([3,5,7,9,15]),
    1,OLpmf,OLrvs,0.75,['OL','n'],cilims,-1)
###########################################
# MAIN
###########################################
    
def makeParTab(dgps):
    D=[['Distribution','Range $c_l$','$c_l^\Delta$ value','$c_l^\Delta$ range','NP label','NP value', 'NP range']]
    for i in range(len(dgps)):
        dgp=dgps[i]
        D.append([DLBLS[i],'$[%d,%d]$'%(dgp.cl[0],dgp.cl[-1]),
        dgp.np1[2],'$[%d,%d]$'%(dgp.np1[0],dgp.np1[-1]),
        '$%s$'%dgp.np2lbl,dgp.np2[2],'$[%d,%d]$'%(dgp.np2[0],dgp.np2[-1])])
    ndarray2latextable(np.array(D))

def makeSuppText(dgps):
    figstr='''\\begin{{figure}}[ht]\n\\centering
\\includegraphics[scale=0.6]{{../code/figures/supplement/{2}{3}}}
\\caption{{{0} evaluated with {1} }}\n\\end{{figure}}\n\\newpage\n''' 
    ft=['Skew','Confidence intervals','Cohen\'s $d$',
        'Two-group tests','Three-group tests','TOST procedure',
        'Bayesian t test']
    dt=['Generalized gamma distribution','Wald distribution',
        'Beta prime distribution','Beta distribution',
        'Beta-binomial distribution',
        'Ordered logistic regression model']
    s=''
    for i in range(len(dgps)):
        s+='''\\section{{{0}}}\n'''.format(dt[i])
        for j in range(1,len(FIGLBLS)):
            s+=figstr.format(ft[j],dt[i],dgps[i].suf,FIGLBLS[j])
            
    s+='''\\section{{{0}}}\n'''.format('ANOVA with largest $c_l^\Delta$')
    s+='''\\begin{figure}[ht]\n\\centering
\\includegraphics[scale=0.6]{../code/figures/supplement/ANOVA42}
\\caption{ANOVA with largest $c_l^\Delta$. The precise value for each distribution is presented as the upper bound in fourth column in table 2 in the published article.}\n\\end{figure}\n\\newpage\n''' 
    print(s)
    
if __name__ == '__main__':
    from sys import version_info
    print('python version: %d.%d.%d'%version_info[:3])
    print('numpy  version: '+np.__version__)
    print('scipy  version: '+__version__)
    from pystan import __version__ as pv
    print('pystan version: '+pv)   
    import warnings
    warnings.filterwarnings("ignore")
    dgps=[gengamma,wald,betaprime,betadgp,betabinom,olrm]
    #makeParTab(dgps)        
    #makeSuppText(dgps) 
    
    
    for dgp in dgps:
        #uncomment to run simulations
        #dgp.compute()
        dgp.plot()
        dgp.makeSupplement()
    print('Creating summary figures')
    plotANOVAall(dgps)
    plotANOVAall(dgps,j=4)
    plotANOVAall(dgps,j=0)
    plotANOVAall(dgps,i=0)
    plotANOVAall(dgps,i=4)
    plotSkew(dgps)
    plotXtypes(S)
    plotFcfe()
    from stanSims import plot as plotStanSims
    plotStanSims(FIGDIR,DPI)
    print('Success')
    
