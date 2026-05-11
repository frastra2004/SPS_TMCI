import h5py

def tree(g, indent=0):
    for key in g.keys():
        item = g[key]
        print("  " * indent + "|-- " + key)
        if isinstance(item, h5py.Group):
            tree(item, indent + 1)


def print_tree(file):
    with h5py.File(file, 'r') as f:
        tree(f)

def open_file(file):
    with h5py.File(file, 'r') as f:

        vert_delta = f['vertical/delta'][:]   
        horiz_delta = f['horizontal/delta'][:]
        horiz_sigma = f['horizontal/sigma'][:]
        vert_sigma = f['vertical/sigma'][:]

    return vert_delta, vert_sigma, horiz_delta, horiz_sigma, 