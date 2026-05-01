
import yaml



def read_yaml(fpath_yaml):
    yaml_conf = None
    with open(fpath_yaml) as f_yaml:
        yaml_conf = yaml.load(f_yaml, Loader=yaml.FullLoader)
    return yaml_conf






