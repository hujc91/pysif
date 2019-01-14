import sys
sys.path.append('../..')

import pysif

tg_solver = pysif.solver('taylor-green-512.ini')
tg_solver.initialization()
tg_solver.solve()
