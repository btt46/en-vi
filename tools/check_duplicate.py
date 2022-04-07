import re
from weakref import ref
import sentencepiece 
import argparse

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i','--input_file', type=str, required=True, dest='input', help='Input file name')
    parser.add_argument('-r','--ref_file', type=str, required=True, dest='ref', help='Reference file name')
    
    args = parser.parse_args()
    input_contents = []
    ref_contents = []

    with open(args.input, 'r', encoding='utf-8') as fp:
        input_contents = fp.readlines()
        fp.close()

    with open(args.ref, 'r', encoding='utf-8') as fp:
        ref_contents = fp.readlines()
        fp.close()

    score = 0
    for line in input_contents:
        if line in ref_contents:
            score += 1

    print(f"Score: {score}")

if __name__ == '__main__':
    main()

