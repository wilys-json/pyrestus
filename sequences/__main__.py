import argparse
from pathlib import Path
from src import (mlcs_leveled_DAG, similarity, list_possible_combinations)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+', type=str, metavar='inputs',
                        help='input strings')
    parser.add_argument('-m', '--measure', default='mlcs', type=str, metavar='measure',
                        help='Regex patterns to look for.')
    parser.add_argument('--similarity-measure', type=str, metavar='similarity',
                        help='simialrity measure to use')

    args = parser.parse_args()


if __name__ == '__main__':
    main()
