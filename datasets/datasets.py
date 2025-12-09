import sklearn.datasets as datasets

def create_two_moons(n_samples, noise=0.15):
    return datasets.make_moons(n_samples, noise=noise)