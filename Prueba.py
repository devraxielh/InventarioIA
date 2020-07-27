import SPCC_Demanda

BD='colombia2'
fecha='2020-07-27'

#EntrenarDemanda(BD,95,'D',28426,11,fecha)#cantidad de datos para calcular la metrica, frecuencia W=semanal D=diario,#fecha de entrenamiento
#EntrenarDemandaALL(BD,95,'D',fecha)#cantidad de datos para calcular la metrica, frecuencia W=semanal D=diario,#fecha de entrenamiento

#forecast=PredecirDemandaAll(BD,28426,11,7)#BD,Cod_Producto,Cod_Centro_Operativo,cantidad_a_predecir
#forecast=PredecirDemanda(BD,28426,11,7)#BD,Cod_Producto,Cod_Centro_Operativo,cantidad_a_predecir

#Lista de productos/centro_operativo que fallaron
#13 productos del Centro Operativo 11: 11058, 15372, 30023, 30025, 30168, 30169, 30170, 30171, 44013, 46233, 46243, 46244, 46412
#10 productos del Centro Operativo 12: 15466, 26841, 30031, 30176, 34494, 34496, 46243, 46244, 46290, 46412,
#6 productos del Centro Operativo 14: 34460, 35630, 46191, 46233, 46243, 46244
#6 productos del Centro Operativo 15: 30028, 32091, 39884, 40808, 46243, 46244
#2 productos del Centro Operativo 18: 30033, 32679
#6 productos del Centro Operativo 19: 16217, 30172, 30178, 35630, 46243, 46244
#5 productos del Centro Operativo 20: 26842, 27483, 30294, 35361, 46243
#6 productos del Centro Operativo 23: 30035, 31903, 35634, 39934, 46191, 46243
#8 productos del Centro Operativo 24: 11058, 30023, 30024, 30025, 30026, 34453, 46243, 46244

Producto = "28426"
Centro_Operativo = "11"

print("\n\n")
#entrenamiento=SPCC_Demanda.EntrenarDemanda(BD,95,'W',Producto,Centro_Operativo,fecha)
#print(entrenamiento)
forecast=SPCC_Demanda.PredecirDemanda(BD,Producto,Centro_Operativo,7,fecha)
print(forecast)
#if(entrenamiento.empty):
   #print("Error sin Datos")
#else:
   #print(entrenamiento[['mae_auto','mae_holt','mae_promedio']].to_dict('records'))
   #print("\n")
   #forecast=SPCC_Demanda.PredecirDemanda(BD,Producto,Centro_Operativo,7,fecha)
   #print(forecast)