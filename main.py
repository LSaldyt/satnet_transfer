
import sys, os
from importlib import import_module
from pathlib import Path
import warnings
from rich.traceback import install
install(show_locals=False)
from rich import print
from rich.pretty import pprint

# By default, run a script in scripts/ dir
def script_selector(script, *args):
    script_file = 'scripts/' + script + '.py'
    if os.path.isfile(script_file):
        module = import_module('scripts.' + script, package='.')
        code = module.run(*args)
        print(f'Finished running script {script_file}!', flush=True)
        if code is not None and code != 0:
            return code
    return 0

def main(args):
    if len(args) == 1:
        name = args[0]
    else:
        name = 'main'
        args.insert(0, name)
    script_file = 'scripts/' + name + '.py'
    if os.path.isfile(script_file):
        script_selector(*args)
    else:
        print(f'Script {script_file} not found!')


if __name__ == '__main__':
    main(sys.argv[1:])
