import ebdba_logic as B,pickle as J,Config as C
def get_trainingData(data):
	A={};D=C.keys
	for B in D:A[B]=[data[B][A-1]for A in C.files_to_create_ebdba_template]
	return A
def get_ebdba_Template(user,user_all_file_data):
	L=f"EBDBA_Template\\{C.DB}\\USER{str(user)}_EBDBA";G=get_trainingData(user_all_file_data);M=B.avg_length(G['X']);D={};E=C.keys
	for A in E:
		H=[]
		for N in G[A]:H.append(B.interpolate_signature(N,M))
		D[A]=H
	I={}
	for A in E:I[A]=B.get_mean(D[A])
	F={}
	for A in E:F[A]=B.dba(I[A],D[A])
	with open(L,'wb')as O:J.dump(F,O)
	return F