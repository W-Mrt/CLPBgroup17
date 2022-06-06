import numpy as np

# a nonlinear function of a 2d array x
def f(x,c=1):
    r=0
    if c==1:
        if x[0]>-20 and x[1]>-40 and x[0]+x[1] < 40:
            r=1
    if c==2:
        if (np.sign(x.sum())*np.sign(x[0]))*np.cos(np.linalg.norm(x)/(2*np.pi))>0:
            r=1
    return r

def create_data(N=4000, B=100, c='a'):
    if c=='a':
        x = (np.random.random((N,2))-0.5)*B
        y = ((x[:,0]>-20) & (x[:,1]>-40) and (x[:,0]+x[:,1] < 40)).astype(int)
    elif c=='b':
        x = (np.random.random((N,2))-0.5)*B
        y = ((np.sign(x.sum(axis=1))*np.sign(x[:,0])) *np.cos(np.linalg.norm(x, axis=1)/(2*np.pi))>0).astype(int)
    elif c=='c':
        r, R = B/5, B/5 +B/10
        rand = np.random.random(N)
        ra = (np.array([r if rand[i] < 0.5 else R for i in range(N)]) + (np.random.random(N)-0.5)*B/10)
        alpha = 2*np.pi*np.random.random(N)
        x = np.empty((N,2))
        x[:,0] = np.cos(alpha)*ra
        x[:,1] = np.sin(alpha)*ra
        y = (rand < 0.5).astype(int)

def filename(s,TYPE=1):
    return "./DATA/"+s+"-for-DNN-"+str(TYPE)+".dat"
