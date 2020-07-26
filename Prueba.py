import SPCC_Demanda

DB='colombia2'
fecha='2020-07-22'

#SPCC_Demanda.EntrenarDemanda(DB,95,'D',28426,11,fecha)
SPCC_Demanda.EntrenarDemandaALL(DB,95,'D',fecha)

#forecast=SPCC_Demanda.PredecirDemandaAll(DB,28426,11,7,fecha)
#forecast=SPCC_Demanda.PredecirDemanda(DB,28426,11,7,fecha)
#print(forecast)