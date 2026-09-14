#!/usr/bin/env python3
import argparse, json
import numpy as np
from scipy.stats import t

def student_t_ci(values, confidence=0.95):
    x=np.asarray(values,float); n=len(x); mean=float(x.mean()); sd=float(x.std(ddof=1)) if n>1 else 0.0
    if n<2: return {'mean':mean,'sd':sd,'low':mean,'high':mean,'n':n}
    half=float(t.ppf((1+confidence)/2,n-1)*sd/np.sqrt(n))
    return {'mean':mean,'sd':sd,'low':mean-half,'high':mean+half,'n':n}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('values',nargs='+',type=float); args=ap.parse_args(); print(json.dumps(student_t_ci(args.values),indent=2))
if __name__=='__main__': main()
