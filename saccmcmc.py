import saccfit as sf
import pymc

if __name__ == '__main__':
    iters = 20000
    burn = 4000
    thin = 4
    save_int = 1000
    db = 'pickle'
    dbfile = 'saccade_fit.pkl'
    
    model = pymc.MCMC(sf, db=db, dbname=dbfile)
    
    model.sample(iter=iters, burn=burn, thin=thin, save_interval=save_int)
    
    m.db.close()
