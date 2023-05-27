import numpy as np

def getRapidity(p4):
    
    return 0.5 * np.log(( p4.energy + p4.pz ) / ( p4.energy - p4.pz ))
    
    
    