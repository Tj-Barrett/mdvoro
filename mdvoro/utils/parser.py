import argparse, os, glob, multiprocessing, sys
import numpy as np
from mdvoro.utils.read_config import read_config
from mdvoro.utils.read_yaml import read_yaml

def parser():
    parser = argparse.ArgumentParser(
        prog='MDvoro', usage='%(prog)s [-f...] [options]',
        description="MDvoro : A Python package for hierarchical multiscale modeling."
    )

    #===========================================================
    #
    # Main Args
    #
    #===========================================================
    group_required = parser.add_argument_group('Required Commands', 'These are the basic MDvoro functions.')
    required = group_required.add_mutually_exclusive_group()

    required.add_argument(
        "-voro", 
        help="Compute voro", 
        action="store_true"
    )
    required.add_argument(
        "-tetgen", 
        help="Create mesh", 
        action="store_true"
    )
    required.add_argument(
        "-test", 
        help="Create mesh", 
        action="store_true"
    )

    args = parser.parse_args()

    # Add obabel and vina locations
    args = read_config(args)

    return args

def sanitize_inputs(args):
    # ===========================================================
    #
    # We need to do os.path.exists(...)
    #
    # ===========================================================

    pargs = parsed_args()

    #===========================================================
    #
    # Applications
    #
    #===========================================================
    if isString(args.alias_tetgen):
        pargs.alias_tetgen=args.alias_tetgen
    else:
        print("- tetgen alias is not specified in the config file, defaulting to tetgen.")
        pargs.alias_tetgen="tetgen"

    if isString(args.alias_voro):
        pargs.alias_voro=args.alias_voro
    else:
        print("- voro++ alias is not specified in the config file, defaulting to voro++.")
        pargs.alias_voro="voro++"

    # ===========================================================
    #
    # Main Run Options
    #
    # ===========================================================

    if args.dockmethod:
        pargs.dockmethod = args.dockmethod
    else:
        print('No docking method specified. This should not happen.')
        exit()

    if args.dock:
        pargs.dock = True
        print(f"+ Docking with {pargs.dockmethod}")
    else:
        pargs.dock = False

    return pargs

class parsed_args:
    def __init__(self):
        # Applications
        self.alias_tetgen = None
        self.alias_voropp = None
