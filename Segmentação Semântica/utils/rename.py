import os

arquivos = os.listdir(os.getcwd())

arquivos = [a for a in arquivos if not(a.endswith('.py'))]

for arq in arquivos:
	inteiro = int(arq.split('.')[0])
	if inteiro < 10:
		novo_nome = '000'+str(inteiro)+'.png'
	elif inteiro < 100:
		novo_nome = '00'+str(inteiro)+'.png'
	else:
		novo_nome = '0'+str(inteiro)+'.png'

	os.rename(arq, novo_nome)