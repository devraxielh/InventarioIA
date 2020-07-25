import os
import pickle
import my_conexion
import pandas as pd
import numpy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from pandasql import sqldf
from datetime import datetime
pysqldf = lambda q: sqldf(q, globals())
pd.options.display.max_columns = None

def EntrenarDemoraProveedor(BD):
    dir_path = os.path.dirname(os.path.abspath("__file__"))    
    nuevaruta = dir_path +'/ModelosEntrenados/' + BD + "/"
    if not os.path.exists(nuevaruta): 
        os.makedirs(nuevaruta)
    conn = my_conexion.getConnection(BD)
    f="where lead_time_real_max_compra<>0 and lead_time_real_max_compra<100 and recepcion_max_fecha_real <>'1/01/1900'"
    sql_compras = "SELECT * FROM compras "+f+""
    
    Compras=pd.read_sql(sql_compras,conn)

    if len(Compras) == 0:
        return 'Error Sin datos'
    
    Compras['fecha_orden_compra'] = pd.to_datetime(Compras['fecha_orden_compra'])
    Compras['dia_orden_compra']=Compras['fecha_orden_compra'].map(lambda x: x.strftime('%d'))
    Compras['mes_orden_compra']=Compras['fecha_orden_compra'].map(lambda x: x.strftime('%m'))
    Compras['ano_orden_compra']=Compras['fecha_orden_compra'].map(lambda x: x.strftime('%Y'))
 
    columnsTitles=[
        'cod_producto',
        'cod_centro_operativo',
        'compra_precio_unitario',
        'lead_time_teorico_compra',
        'dia_orden_compra',
        'mes_orden_compra',
        'ano_orden_compra',
        'lead_time_real_max_compra']
    
    Compras=Compras.reindex(columns=columnsTitles)
    train, test = train_test_split(Compras, test_size = 0.2)
    
    colsnames = Compras.columns.values.tolist()
    predictors = colsnames[0:len(colsnames)-1]
    target = colsnames[len(colsnames)-1]
    
    X = Compras[predictors]
    y = Compras[target]

    X_train = train[predictors]
    y_train = train[target]

    X_test = test[predictors]
    y_test = test[target]
    
    msg=''
    try:
        forest_metricas = RandomForestRegressor(criterion='mse',n_estimators=200,oob_score=True)
        forest_metricas.fit(X_train,y_train).score(X_test,y_test)
        forest_produccion = RandomForestRegressor(criterion='mse',n_estimators=200,oob_score=True)
        forest_produccion.fit(X,y)
        ltc=X_test['lead_time_teorico_compra'].values
        y_pred = forest_metricas.predict(X_test)
        p,t=testeo(y_test,y_pred,ltc) 
        p=round(p,3)*100
        t=round(t,3)*100
        filename = nuevaruta +'DemoraProveedor.pkl'
        
        metricas = {
            'MapePrediccion': [p],
            'MapeTeorico': [t]
        }

        df = pd.DataFrame(metricas, columns= ['MapePrediccion','MapeTeorico'])
        df.to_csv(nuevaruta+'metricas_LeadTime.csv', index = False, header=True)
        pickle.dump(forest_produccion,open(filename,'wb'))
        msg=filename
    except:
        msg='Error'
    return msg

def testeo(y_test,y_pred,ltc):
    mape_predicho = numpy.mean(abs(y_test-y_pred)/y_test)
    mape_teorico = numpy.mean(abs(y_test-ltc)/y_test)
    return mape_predicho,mape_teorico

def PredecirDemoraProveedor(BD,Cod_Producto,Cod_Centro_Operativo,fecha):
    try:
        dir_path = os.path.dirname(os.path.realpath("__file__"))
        nuevaruta = dir_path +'/ModelosEntrenados/' + BD + "/"
        filename = nuevaruta +'DemoraProveedor.pkl'
        loaded_model = pickle.load(open(filename,'rb'))
        conn = my_conexion.getConnection(BD)
        sql = "SELECT compra_precio_unitario,lead_time_teorico_compra FROM compras where cod_producto="+str(Cod_Producto)+" and cod_centro_operativo="+str(Cod_Centro_Operativo)+" order by fecha_orden_compra DESC Limit 1"
        datos=pd.read_sql(sql,conn)

        if len(datos) == 0:
            return 'Error Sin datos'

        compra_precio_unitario=str(datos['compra_precio_unitario'][0])
        lead_time_teorico_compra=str(datos['lead_time_teorico_compra'][0])
        fecha = datetime.strptime(fecha, "%Y-%m-%d")
        mes=fecha.month
        dia=fecha.day
        ano=fecha.year

        columnsTitles=[
            'cod_producto',
            'cod_centro_operativo',
            'compra_precio_unitario',
            'lead_time_teorico_compra',
            'dia_orden_compra',
            'mes_orden_compra',
            'ano_orden_compra',
        ]

        imput = pd.DataFrame(columns=columnsTitles)
        imput.loc[len(imput)]=[Cod_Producto,Cod_Centro_Operativo,compra_precio_unitario,lead_time_teorico_compra,dia,mes,ano]
        result = loaded_model.predict(imput.values)

        metricas=pd.read_csv(nuevaruta+'metricas_LeadTime.csv')

        if(metricas['MapePrediccion'][0]<metricas['MapeTeorico'][0]):
            lista = [
                        ["Prediccion",result[0]],
                        ["Teorico",lead_time_teorico_compra],
                        ["MapePrediccion",metricas['MapePrediccion'][0]],
                        ["MapeLeadTime",metricas['MapeTeorico'][0]]
                    ]
        else:
            lista = [
                        ["Teorico",lead_time_teorico_compra],
                        ["Prediccion",result[0]],
                        ["MapePrediccion",metricas['MapePrediccion'][0]],
                        ["MapeLeadTime",metricas['MapeTeorico'][0]]
                    ]
        return lista
    except Exception as e:
        print(str(e))

BD='colombia2'
prediccion=PredecirDemoraProveedor(BD,28431,23,'2019-07-18')
print(prediccion)
#[['Prediccion', 16.63],
# ['Teorico', '10'],
# ['MapePrediccion', 31.5],
# ['MapeLeadTime', 65.3]]
#EntrenarDemoraProveedor(BD)