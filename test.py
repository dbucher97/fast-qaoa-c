from fastqaoa.ctypes import Diagonals

keys = [0b01, 0b10, 0b11]

vals = [-1, -1, 2]

diag = Diagonals.brute_force(2, keys, vals)

diag.print()
