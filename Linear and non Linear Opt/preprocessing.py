import argparse

import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nation', type='str',
                        help='Nation of the instrument to preprocess')
    parser.add_argument('-i', '--instument', type='str',
                        help='Type of financial instrument to parse')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    if args.nation == 'DK':
        if args.instument == 'stock':
            pass