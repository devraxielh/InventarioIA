import SPCC_Demanda


fecha='2020-07-22'
#SPCC_Demanda.EntrenarDemanda(95,'D',28426,11,fecha)
#SPCC_Demanda.EntrenarDemandaALL(95,'D',fecha)

#forecast=SPCC_Demanda.PredecirDemandaAll(28426,11,7)
forecast=SPCC_Demanda.PredecirDemanda(28426,11,7)
print(forecast)