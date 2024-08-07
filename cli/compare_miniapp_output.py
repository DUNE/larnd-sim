#!/usr/bin/env python3

import argparse
import pickle

import numpy as np


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('file1')
    ap.add_argument('file2')
    args = ap.parse_args()

    d1 = load(args.file1)
    d2 = load(args.file2)

    for k in d1:
        if np.all(d1[k] == d2[k]):
            print(f'{k} looks good')
        else:
            print(f'{k} looks bad!!!')


if __name__ == '__main__':
    main()
