#!/usr/bin/env python3

from mdvoro.parser import parser
from mdvoro.compute_voro import compute_voro
from mdvoro.ovito_writer import ovito_writer
from mdvoro.triangle_face import triangle_face
from mdvoro.tetgen import tetgen

def splash():
    return "\n=====================================================================\n\tMDvoro v2025.09\n\tDOI:\n\tBarrett, Muftu, and Minus.\n\n\tIssues? : barrett@northeastern.edu\n=====================================================================\n"


def main():
    args = parser()
    ################################################################
    #
    # init
    #
    ################################################################
    print(splash())




if __name__ == "__main__":
    main()
