import re
import sys
import argparse
from pathlib import Path

def _find_files(folder):
    files = []
    if all(list(map(Path.is_file, list(folder.iterdir())))):
        return list(folder.iterdir())

    for child in folder.iterdir():
        files += _find_files(child)

    return files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pattern', type=str, metavar='pattern',
                        help='Regex patterns to look for.')
    parser.add_argument('folder', type=str, metavar='folder',
                        help='Folder containing files to unify.')
    parser.add_argument('-r', '--recursive', action='store_true', default=False)

    args = parser.parse_args()

    find = lambda x : re.findall(args.pattern, x)
    folder = Path(args.folder)

    assert folder.is_dir(), f"Cannot find folder: {args.folder}."

    if args.recursive:
        files = _find_files(folder)
    else:
        files = folder.iterdir()

    count = 0
    for file in files:
        if file.name[0] != '.':
            found = find(str(file))
            if found:
                file.rename(file.parent / find(str(file))[0])
                count += 1

    print(f'Renamed {count} files.')

if __name__ == '__main__':
    main()
