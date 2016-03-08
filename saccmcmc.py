import tmfit2 as tmf2
import pymc

if __name__ == '__main__':
    iters = 10000
    burn = 1000
    thin = 1
    save_int = 2
    db = 'pickle'
    dbfile = 'tm_fit_srs_rerun2.pkl'
    
    model = pymc.MCMC(tmf2, db=db, dbname=dbfile)
    
    model.sample(iter=iters, burn=burn, thin=thin, save_interval=save_int)
    
    model.db.close()
