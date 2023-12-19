#!/usr/bin/env python

from larndsim.config import list_config_keys

print('\nlarnd-sim configuration keys:\n')
for k in list_config_keys():
	print('   ',k)
print()