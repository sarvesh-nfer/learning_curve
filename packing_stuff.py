import pkg_resources
import subprocess,sys,os

REQUIRED = {
  'pandas', 'sqlite3', 'sys', 'os', 'glob', 
  'xlrd', 'plotly', 'seaborn', 'matplotlib'
}

installed = {pkg.key for pkg in pkg_resources.working_set}
missing = REQUIRED - installed

if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)