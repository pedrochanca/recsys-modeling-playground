import itertools


def param_comb(config, is_tune: bool):

    if is_tune:

        keys, values = zip(*config.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    else:
        combinations = [config]

    return combinations
