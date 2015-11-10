import saccfit as sf
import pymc

if __name__ == '__main__':
    iters = 60000
    burn = 10000
    thin = 5
    save_int = 10000
    db = 'pickle'
    dbfile = 'saccade_fit.pkl'
    
    model = pymc.MCMC(sf, db=db, dbname=dbfile)
    
    model.sample(iter=iters, burn=burn, thin=thin, save_interval=save_int)
    
    m.db.close()
