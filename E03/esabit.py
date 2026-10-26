nn1 = input( 'inserisci primo valore ', )

nn2 = input( 'inserisci secondo  valore ', )

n1 = int(nn1)
n2 = int(nn2)

print( bin(n1))
print( bin(n2))
print( hex(n1))
print( hex(n2))


print('bitwise OR / AND ...')
print ("----------------------")
print ("    primo valore   : {:06b}".format(n1) )
print ("  secondo valore   : {:06b}".format(n2) )
print ("----------------------")
print (" OR : {:06b}".format(n1|n2) )
print ("AND : {:06b}".format(n1&n2) )

print ("----------------------")
print ("    primo valore (hex)   : {:04x}".format(n1) )
print ("  secondo valore (hex)   : {:04x}".format(n2) )
print ("----------------------")
print (" OR (hex) : {:04x}".format(n1|n2) )
print ("AND (hex) : {:04x}".format(n1&n2) )
