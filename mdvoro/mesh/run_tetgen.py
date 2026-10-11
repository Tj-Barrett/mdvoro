'''
Author : Thomas J. Barrett
Advisor : Marilyn L. Minus

Northeastern University

See the below papers for methods/ code / more information

--------------------------------------------------------------------------------
TetGen
--------------------------------------------------------------------------------


================================================================================
TetGen
================================================================================

INPUTS:

OUTPUTS:


'''
import os, subprocess
from mdvoro.utils.read_config import read_config

def run_tetgen(filename, tetcmds):
    args = read_config()
    cmd = args.alias_tetgen
    p = subprocess.run([cmd,tetcmds,filename])
