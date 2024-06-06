import os

# This function prints the directory structure of the current directory

def print_directory_structure(directory):
    def print_tree(root, indent=''):
        items = [item for item in os.listdir(root) if item != '.git']
        for i, item in enumerate(sorted(items)):
            path = os.path.join(root, item)
            is_last = i == len(items) - 1
            marker = '└── ' if is_last else '├── '
            print(indent + marker + item)
            if os.path.isdir(path):
                next_indent = indent + ('    ' if is_last else '│   ')
                print_tree(path, next_indent)

    print_tree(directory)

print_directory_structure('.')
