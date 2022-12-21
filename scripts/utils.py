'''
Defines utility functions.
Author: Daniel von Eschwege
Date:   2 November 2022
'''

import os
import re
import time
import json
import pickle
import shutil
import logging
from pathlib import Path
from multiprocessing import Process
from scripts.const import *


def agg(dictio, key, value):
    '''
    Set key in dictionary to the value if it doesn't exist,
    else sum the value with the existing value.
    '''

    if key in dictio.keys():
        dictio[key] += value
    else:
        dictio[key] = value


def mkdirs(path):
    '''
    Create directories recursively,
    do nothing for directories which
    already exist.
    '''

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def pl(text):
    '''
    Print & Log a text string.
    '''

    print(text)
    logging.info(text)


def replaceDir(src, dst):
    '''
    Remove dst if exists.
    Replace dst with src.
    '''
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(f'{src}/', f'{dst}/')


def addDir(src, dst):
    '''
    Add src to dst with timestamp postfix.
    '''

    postfix = f'{int(time.time())}'
    shutil.copytree(f'{src}/', f'{dst}--{postfix}/')


def rm(path):
    '''
    Delete a file if it exists.    
    '''

    if os.path.exists(path):
        os.remove(path)


def lpkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    return data


def wpkl(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=5)


def ljson(path):
    with open(path, 'r') as f:
        data = json.load(f)
    
    return data


def wjson(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

    
def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.
 
    Required arguments:
    l -- The iterable to be sorted.
 
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)


def multiProc(target, args, nProcesses, experiments):
    '''
    Runs a function on multiple cores.
    '''
    
    # creating processes
    nExperiments = len(experiments)
    processes = {}
    tExperiments = [nExperiments // nProcesses +
                    (1 if x < nExperiments % nProcesses else 0)
                    for x in range (nProcesses)]

    n = 0
    for p, pE in zip(range(nProcesses), tExperiments):
        argsP = args.copy()
        argsP.append(experiments[n:n+pE])
        argsP.append(p)
        processes[f'p{p}'] = Process(target=target, args=tuple(argsP))
        n += pE

    # start processes
    for p in processes.values():
        p.start()

    # wait for processes to finish
    for p in processes.values():
        p.join()


def fi(x):
    '''
    Converts a float to int if it contains
    nothing after the decimal point.
    '''
    
    assert x%1==0, 'The value should be an integer'
    return int(x)