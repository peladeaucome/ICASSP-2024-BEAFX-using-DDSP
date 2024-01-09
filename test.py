import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train', default=True, action=argparse.BooleanOptionalAction)

args = parser.parse_args()
print(args.train)