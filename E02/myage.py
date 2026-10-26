from datetime import datetime, timedelta

datetime.now()

datenow = datetime.now()

print(datenow.year)
print(datenow.hour)
print(datenow.second)



compleanno = input('inserisci data di nascita (gg-mm-aaaa) (ho:mi:se): ' )


mydate = datetime.strptime(compleanno, "%d-%m-%Y %H:%M:%S")

print('date:',       mydate       )
print('year:',  mydate.year  )
print('second:',mydate.second)

timediff = datenow - mydate

tots = timediff.total_seconds()

totd = tots/(3600*24)

toty = datenow.year - mydate.year


age_dict = {'Anni' : toty, 'Giorni' : totd, 'Secondi' : tots}

for k in age_dict:
    print('{: <10} {:f}'.format(k, age_dict[k]))
