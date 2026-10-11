import os
import yaml 

def read_config(args=None):
    if args is None:
        args = config_args()
    # ROOT_DIR will be utils, go up one
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    conf = f"{ROOT_DIR}/config.yaml"
    if os.path.isfile(conf):
        with open(conf) as stream:
            yamlin = yaml.safe_load(stream)

            if yamlin["tetgen"]:
                args.alias_tetgen = yamlin["tetgen"]
            
            if yamlin["voro++"]:
                args.alias_voropp = yamlin["voro++"]

        return args

    else:
        print(
            f"No config file {conf} available. We need one to direct the scripts properly. Exiting... "
        )
        quit()

class config_args:
    def __init__(self):
        self.alias_tetgen=None
        self.alias_voropp=None