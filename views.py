#==================================================
#   00. Header
#==================================================

import os

current = os.getcwd()
utils_directory = 'myapp/utils/views/'
os.chdir('/home/codespace/Roos/myproject/myapp/utils/views/')

#===================================================
#   01. Main
#===================================================


from conf import *
from supabase import *
from navigation import *

os.chdir(current)
