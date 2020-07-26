import os
import my_conexion
import pmdarima as pm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from math import sqrt
from datetime import datetime,timedelta
from sklearn import metrics
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
pd.options.display.max_columns = None

def EntrenarDemandaALL(BD,cantidad_particion,frecuencia,fecha):
    sql_pedidos = "SELECT cod_centro_operativo,cod_producto FROM pedidos group by cod_centro_operativo,cod_producto  limit 3"
    conn = my_conexion.getConnection(BD)
    Pedidos=pd.read_sql(sql_pedidos,conn)
    Resultado=pd.DataFrame()
    Resultado["cod_producto"] = range(0,len(Pedidos))
    Resultado["cod_centro_operativo"] = range(0,len(Pedidos))
    Resultado["cantidad_de_datos"] = range(0,len(Pedidos))
    
    df = pd.DataFrame(columns=('centro_operativo', 'producto','cantidad_datos', 'mae_auto', 'mae_holt', 'mae_promedio'))
    
    for i in range(len(Pedidos)):
        cco=Pedidos['cod_centro_operativo'][i]
        cp=Pedidos['cod_producto'][i]
        VerficarDemandaALL(int(cco),int(cp),int(cantidad_particion),df,frecuencia,conn,fecha,BD)
    return df

def EntrenarDemanda(BD,cantidad_particion,frecuencia,cp,cco,fecha):
    
    f="where cod_centro_operativo="+str(cco)+" and cod_producto="+str(cp)+" group by cod_centro_operativo,cod_producto"
    sql_pedidos = "SELECT cod_centro_operativo,cod_producto FROM pedidos "+f+""
    conn = my_conexion.getConnection(BD)
    Pedidos=pd.read_sql(sql_pedidos,conn)
    Resultado=pd.DataFrame()
    Resultado["cod_producto"] = range(0,len(Pedidos))
    Resultado["cod_centro_operativo"] = range(0,len(Pedidos))
    Resultado["cantidad_de_datos"] = range(0,len(Pedidos))
    
    df = pd.DataFrame(columns=('centro_operativo', 'producto','cantidad_datos', 'mae_auto', 'mae_holt', 'mae_promedio'))
    
    for i in range(len(Pedidos)):
        cco=Pedidos['cod_centro_operativo'][i]
        cp=Pedidos['cod_producto'][i]
        VerficarDemanda(int(cco),int(cp),int(cantidad_particion),df,frecuencia,conn,fecha,BD)
    return df

def VerficarDemandaALL(cco,cp,cantidad_particion,df,frecuencia,conn,fecha,BD):
        f="where cod_centro_operativo="+str(cco)+" and cod_producto="+str(cp)+" And fecha_pedido<='"+(fecha)+"' order by fecha_pedido ASC"
        #f="where cod_centro_operativo="+str(cco)+" and cod_producto="+str(cp)+" order by fecha_pedido ASC"
        sql_pedidos = "SELECT fecha_pedido,pedido_cantidad FROM pedidos "+f+""
        Pedidos=pd.read_sql(sql_pedidos,conn)
        columnsTitles=['fecha_pedido','pedido_cantidad']
        Pedidos=Pedidos.reindex(columns=columnsTitles)
        Pedidos['fecha_pedido'] = pd.to_datetime(Pedidos['fecha_pedido'])
        #fechas = pd.date_range(Pedidos['fecha_pedido'].min(),Pedidos['fecha_pedido'].max())
        fechas = pd.date_range(Pedidos['fecha_pedido'].min(),fecha)
        fechas = fechas.strftime("%Y-%m-%d")
        fechas = pd.DataFrame(fechas)
        fechas = fechas.rename(columns = {0:'fecha_pedido'})
        fechas['fecha_pedido'] = pd.to_datetime(fechas['fecha_pedido'])    
        pedidos = pd.merge(fechas,Pedidos, on="fecha_pedido", how="outer")
        pedidos = pedidos.fillna(0)
        pedidos.reset_index(drop=True, inplace=True)
        pedidos.index = pedidos['fecha_pedido']
        columnsTitles=['pedido_cantidad']
        pedidos=pedidos.reindex(columns=columnsTitles)    
        pedidos=pedidos.resample(frecuencia).sum() 
        
        cantidad_d_e=int(cantidad_particion)/100
        
        dir_path = os.path.dirname(os.path.abspath("__file__")) 
        nuevaruta = dir_path +'/ModelosEntrenados/' + BD + "/demanda/"
        
        if not os.path.exists(nuevaruta): 
            os.makedirs(nuevaruta)
        
        auto = AutoArima(pedidos,cantidad_d_e,nuevaruta,cco,cp)
        holt = HoltWinter(pedidos,cantidad_d_e,nuevaruta,cco,cp)
        
        #promedio=pedidos[round(cantidad_d_e*(len(pedidos))):]
        
        mae_promedio=Promedio(pedidos,cantidad_d_e,nuevaruta,cco,cp,fecha)
                
        df.loc[len(df)]=[cco,cp,len(Pedidos),auto,holt,mae_promedio]
        df = df.astype({"centro_operativo": int, "producto": int, "cantidad_datos": int})
        df.to_csv(nuevaruta+'metricas_Demanda.csv', index = False, header=True)
        return "Entrenado..."

def VerficarDemanda(cco,cp,cantidad_particion,df,frecuencia,conn,fecha,BD):
        f="where cod_centro_operativo="+str(cco)+" and cod_producto="+str(cp)+" And fecha_pedido<='"+(fecha)+"' order by fecha_pedido ASC"
        #f="where cod_centro_operativo="+str(cco)+" and cod_producto="+str(cp)+" order by fecha_pedido ASC"
        sql_pedidos = "SELECT fecha_pedido,pedido_cantidad FROM pedidos "+f+""
        Pedidos=pd.read_sql(sql_pedidos,conn)
        columnsTitles=['fecha_pedido','pedido_cantidad']
        Pedidos=Pedidos.reindex(columns=columnsTitles)
        Pedidos['fecha_pedido'] = pd.to_datetime(Pedidos['fecha_pedido'])
        fechas = pd.date_range(Pedidos['fecha_pedido'].min(),fecha)
        fechas = fechas.strftime("%Y-%m-%d")
        fechas = pd.DataFrame(fechas)
        fechas = fechas.rename(columns = {0:'fecha_pedido'})
        fechas['fecha_pedido'] = pd.to_datetime(fechas['fecha_pedido'])    
        pedidos = pd.merge(fechas,Pedidos, on="fecha_pedido", how="outer")
        pedidos = pedidos.fillna(0)
        pedidos.reset_index(drop=True, inplace=True)
        pedidos.index = pedidos['fecha_pedido']
        columnsTitles=['pedido_cantidad']
        pedidos=pedidos.reindex(columns=columnsTitles)    
        pedidos=pedidos.resample(frecuencia).sum() 
        
        cantidad_d_e=int(cantidad_particion)/100
        
        dir_path = os.path.dirname(os.path.abspath("__file__")) 
        nuevaruta = dir_path +'/ModelosEntrenados/' + BD + "/demanda/"
        
        if not os.path.exists(nuevaruta): 
            os.makedirs(nuevaruta)
        
        auto = AutoArima(pedidos,cantidad_d_e,nuevaruta,cco,cp)
        holt = HoltWinter(pedidos,cantidad_d_e,nuevaruta,cco,cp)
        
        promedio=pedidos[round(cantidad_d_e*(len(pedidos))):]
        
        mae_promedio=Promedio(pedidos,cantidad_d_e,nuevaruta,cco,cp,fecha)
                
        df.loc[len(df)]=[cco,cp,len(Pedidos),auto,holt,mae_promedio]
        df = df.astype({"centro_operativo": int, "producto": int, "cantidad_datos": int})
        df.to_csv(nuevaruta+'metricas_Demanda_'+str(cp)+'_'+str(cco)+'.csv', index = False, header=True)
        return "Entrenado..."

def AutoArima(pedidos,cantidad_d_e,nuevaruta,cco,cp):
    try:
        train = pedidos[:round(cantidad_d_e*(len(pedidos)))]
        valid = pedidos[round(cantidad_d_e*(len(pedidos))):]
        
        AutoArima = pm.auto_arima(train,error_action='ignore')
        AutoArima.fit(train)
        
        forecast = AutoArima.predict(n_periods=len(valid))
        mea = metrics.mean_absolute_error(valid,forecast)
        auto=mea
        
        AutoArimaProduccion = pm.auto_arima(pedidos,error_action='ignore')
        AutoArimaProduccion.fit(pedidos)
        
        filename = nuevaruta+'demanda_auto_%s_%s.pkl' % (cp,cco)
        pickle.dump(AutoArimaProduccion,open(filename,'wb'))
    
    except Exception as e: 
        auto = "error"
    return auto

def HoltWinter(pedidos,cantidad_d_e,nuevaruta,cco,cp):
    try:
        train = pedidos[:round(cantidad_d_e*(len(pedidos)))]
        valid = pedidos[round(cantidad_d_e*(len(pedidos))):]
        HoltWinter = ExponentialSmoothing(train,trend='add',seasonal='add').fit() 
        forecast = HoltWinter.forecast(len(valid))
        forecast.reset_index(drop=True, inplace=True)
        mea = metrics.mean_absolute_error(valid,forecast)
        ho=mea
        HoltWinterProduccion = ExponentialSmoothing(pedidos,trend='add',seasonal='add').fit() 
        filename = nuevaruta+'demanda_holt_%s_%s.pkl' % (cp,cco)
        pickle.dump(HoltWinterProduccion,open(filename,'wb')) 
    except Exception as e:
        ho = "error"
    return ho

def Promedio(pedidos,cantidad_d_e,nuevaruta,cco,cp,fecha):
    fecha=datetime.strptime(fecha, '%Y-%m-%d').date()
    valid = pedidos[round(cantidad_d_e*(len(pedidos))):]
    if(len(valid)==0):
        mea = 9999999999
    else:
        Promedio=pd.DataFrame()
        Promedio["valor"] = range(0,len(valid))
        Promedio["valor"] = valid.values
        Promedio["fecha"] = valid.index  
        
        fechas = pd.date_range(Promedio["fecha"].min(),fecha)
        fechas = fechas.strftime("%Y-%m-%d")
        fechas = pd.DataFrame(fechas)
        fechas = fechas.rename(columns = {0:'fecha'})
        fechas['fecha'] = pd.to_datetime(fechas['fecha'])
        
        pedidos = pd.merge(fechas,Promedio, on="fecha", how="outer")
        pedidos = pedidos.fillna(0)
        pedidos.reset_index(drop=True, inplace=True)
        pedidos.index = pedidos['fecha']

        Promedio["promedio"] = pedidos.loc[fecha+timedelta(days=-15):str(fecha)].mean().values[0]
        mea = metrics.mean_absolute_error(valid,Promedio["promedio"])
        
    return mea
    
def PredecirDemanda(BD,Cod_Producto,Cod_Centro_Operativo,cantidad_a_predecir,fecha):
    dir_path = os.path.dirname(os.path.abspath("__file__")) 
    nuevaruta = dir_path +'/ModelosEntrenados/' + BD + "/demanda/"
    try:
       metricas=pd.read_csv(nuevaruta+'metricas_Demanda_'+str(Cod_Producto)+'_'+str(Cod_Centro_Operativo)+'.csv')
    except Exception as e:
        return "Error: " + str(e) 
    metricas=metricas[(metricas['producto']==Cod_Producto)&(metricas['centro_operativo']==Cod_Centro_Operativo)]
    columnsTitles=['mae_auto','mae_holt','mae_promedio',]
    imput = pd.DataFrame(columns=columnsTitles)
    imput.loc[len(imput)]=[metricas['mae_auto'],metricas['mae_holt'],metricas['mae_promedio']] 
    
    op=np.argmin(metricas.iloc[:,3:6].values)
    
    if(str(metricas.iloc[:,3:6].values[0][2])=="error valid 0"):
        conn = my_conexion.getConnection(BD)
        f="where cod_centro_operativo="+str(Cod_Centro_Operativo)+" and cod_producto="+str(Cod_Producto)+""
        sql="select round(avg(pedido_cantidad)) as pedido_cantidad from pedidos "+f+""    
        promedio=pd.read_sql(sql,conn)
        datos=pd.DataFrame()
        datos["valor"] = range(0,cantidad_a_predecir)
        datos["valor"] = round(promedio['pedido_cantidad'].values[0])
        
        prediccion=datos["valor"].values
    else:
        if(op==0):
            prediccion = forecastAuto(Cod_Producto,Cod_Centro_Operativo,cantidad_a_predecir,BD)
        if(op==1):
            prediccion = forecastHol(Cod_Producto,Cod_Centro_Operativo,cantidad_a_predecir,BD)
        if(op==2):
            prediccion = forecastProm(Cod_Producto,Cod_Centro_Operativo,cantidad_a_predecir,BD,fecha)
    
    return prediccion

def PredecirDemandaAll(BD,Cod_Producto,Cod_Centro_Operativo,cantidad_a_predecir,fecha):
    dir_path = os.path.dirname(os.path.abspath("__file__")) 
    nuevaruta = dir_path +'/ModelosEntrenados/' + BD + "/demanda/"
    metricas=pd.read_csv(nuevaruta+'metricas_Demanda.csv')
    metricas=metricas[(metricas['producto']==Cod_Producto)&(metricas['centro_operativo']==Cod_Centro_Operativo)]
    metricas = metricas.astype({"mae_auto": float,"mae_holt": float,"mae_promedio": float})

    op=np.argmin(metricas.iloc[:,3:6].values)

    if(metricas.iloc[:,3:6].values[0][2]==9999999999):
        conn = my_conexion.getConnection(BD)
        f="where cod_centro_operativo="+str(Cod_Centro_Operativo)+" and cod_producto="+str(Cod_Producto)+""
        sql="select round(avg(pedido_cantidad)) as pedido_cantidad from pedidos "+f+""    
        promedio=pd.read_sql(sql,conn)
        datos=pd.DataFrame()
        datos["valor"] = range(0,cantidad_a_predecir)
        datos["valor"] = round(promedio['pedido_cantidad'].values[0])
        prediccion=datos["valor"].values
    else:
        if(op==0):
            prediccion = forecastAuto(Cod_Producto,Cod_Centro_Operativo,cantidad_a_predecir,BD)
        if(op==1):
            prediccion = forecastHol(Cod_Producto,Cod_Centro_Operativo,cantidad_a_predecir,BD)
        if(op==2):
            prediccion = forecastProm(Cod_Producto,Cod_Centro_Operativo,cantidad_a_predecir,BD,fecha)
    
    return prediccion

def forecastAuto(cp,ccp,c,BD):
    try:
        dir_path = os.path.dirname(os.path.realpath("__file__"))
        nuevaruta = dir_path +'/ModelosEntrenados/' + BD + "/demanda/"
        filename = nuevaruta +'demanda_auto_'+str(cp)+'_'+str(ccp)+'.pkl'
        loaded_model = pickle.load(open(filename,'rb'))
        forecast=loaded_model.predict(n_periods=c)
    except Exception as e:
        print(str(e))
        
    return (forecast)

def forecastHol(cp,ccp,c,BD):
    try:
        dir_path = os.path.dirname(os.path.realpath("__file__"))
        nuevaruta = dir_path +'/ModelosEntrenados/' + BD + "/demanda/"
        filename = nuevaruta +'demanda_holt_'+str(cp)+'_'+str(ccp)+'.pkl'
        loaded_model = pickle.load(open(filename,'rb'))
        forecast=loaded_model.forecast(c)
    except Exception as e:
        print(str(e)) 
        
    return (forecast.values)

def forecastProm(cp,ccp,c,BD,fecha):
    conn = my_conexion.getConnection(BD)
    f="where cod_centro_operativo="+str(ccp)+" and cod_producto="+str(cp)+" AND fecha_pedido BETWEEN date_add('"+fecha+"', INTERVAL -15 DAY) AND '"+fecha+"'"
    sql="select round(avg(pedido_cantidad)) as pedido_cantidad from pedidos "+f+""    
    promedio=pd.read_sql(sql,conn)

    forecast=pd.DataFrame()
    forecast["c"] = range(0,c)
    forecast["forecast"] = range(0, c)

    if(promedio['pedido_cantidad'].values[0]==None):
        forecast["forecast"]=0
    else:
        forecast["forecast"]=promedio['pedido_cantidad'].values[0]
        
    return (forecast["forecast"].values)