import argparse

def main(config):
    
    ...

def get_args() -> dict:

    parser = argparse.ArgumentParser(description='RL Portfolio Manager')

    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    return training_config


if __name__ == "__main__":
    main(get_args())