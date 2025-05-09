C=len
import numpy as A
from fastdtw import fastdtw as H

def avg_length(lst):
	A=0;sum=0
	for B in lst:sum=sum+C(B);A=A+1
	return sum/A
def interpolate_signature(array,target_length):B=array;B=A.asarray(B,dtype=float);D=A.linspace(0,1,C(B));E=A.linspace(0,1,int(target_length));return A.interp(E,D,B)
def get_mean(lst_of_array):B=A.mean(lst_of_array,axis=0);return B
def custom_distance(a,b):return abs(a-b)
def dba(mean,lst_of_array):
	B=mean;B=A.asarray(B).tolist();E=A.zeros((C(B),1));F=A.zeros(C(B))
	for D in lst_of_array:
		D=A.asarray(D).tolist();M,J=H(B,D,dist=custom_distance)
		for(G,K)in J:E[G]+=D[K];F[G]+=1
	L=A.divide(E,F[:,A.newaxis],out=A.zeros_like(E),where=F[:,A.newaxis]!=0);return L.flatten()