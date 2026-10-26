lgs = ['lun', 'mar', 'mer', 'gio', 'ven', 'sab', 'dom']

lgo = ['gio', 'ven', 'sab', 'dom', 'lun', 'mar', 'mer']

l3 = [ 'gio', 'ven', 'sab']


lsoto = lgo*4 + l3

oct_dict = {1 : 'gio', 2 : 'ver', 3 : 'sab', 4 : 'dom', 5 : 'lun', 6 : 'mar', 7 : 'mer', 8 : 'gio', 9 : 'ver', 10 : 'sab', 11 : 'dom', 12 : 'lun', 13 : 'mar', 14 : 'mer', 15 : 'gio', 16  : 'ver', 17 : 'sab', 18 : 'dom', 19 : 'lun', 20 : 'mar', 21 : 'mer', 22 : 'gio', 23 : 'ver', 24 : 'sab', 25 : 'dom', 26 : 'lun', 27 : 'mar', 28 : 'mer', 29 : 'gio', 30 : 'ven', 31 : 'sab'}


for k in oct_dict:
    print('{: <10} {:02d}'.format(oct_dict[k], k))
