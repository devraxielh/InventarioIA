import pymysql.cursors

def getConnection(DB='colombia2'):
    connection = pymysql.connect(host='127.0.0.1',
                             port=3306,
                             user='root',
                             password='root',
                             db=DB,
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

    return connection


