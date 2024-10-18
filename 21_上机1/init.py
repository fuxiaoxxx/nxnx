import pymysql.cursors

connect = pymysql.Connect(
    host = 'localhost',
    port = 3306,
    user = 'root',
    passwd = '123456',
    db = 'school',
    charset = 'utf8'
)

cursor = connect.cursor()

cursor.execute("DROP TABLE IF EXISTS student")
cursor.execute("DROP TABLE IF EXISTS teacher")
cursor.execute("DROP TABLE IF EXISTS course")
cursor.execute("DROP TABLE IF EXISTS pick_up")
cursor.execute("DROP TABLE IF EXISTS student_score")

sql_student = '''
CREATE TABLE student(
  Sno varchar(5) ,
  Sname varchar(255) ,
  Ssex varchar(255) ,
  E_mail varchar(100),
  Phone_num varchar(255) ,
  Password varchar(255) ,
  PRIMARY KEY (Sno) USING BTREE
) 
'''

sql_teacher = '''
CREATE TABLE teacher(
  Tno varchar(5),
  Tname varchar(255) ,
  Tsex varchar(255) ,
  E_mail varchar(100),
  courses varchar(255) ,
  Password varchar(255) ,
  PRIMARY KEY (Tno) USING BTREE
) 
'''

sql_course = '''
CREATE TABLE course(
  CID varchar(5),
  Cname varchar(255) ,
  Cteacher varchar(255)
  )
'''

sql_pick_up = '''
CREATE TABLE pick_up(
  Sname varchar(255),
  Sno varchar(5) ,
  CID varchar(5) ,
  Cname varchar(100),
  teacher varchar(255) 
  )
'''

sql_student_score = '''
CREATE TABLE student_score(
  Sname varchar(255),
  Sno varchar(5) ,
  CID varchar(5) ,
  Cname varchar(100),
  teacher varchar(255), 
  score int(4)
  )
'''

cursor.execute(sql_student)
cursor.execute(sql_teacher)
cursor.execute(sql_course)
cursor.execute(sql_pick_up)
cursor.execute(sql_student_score)

sql_insert_student = ["INSERT INTO student VALUES ('10001', 'Jack', 'male', '10001@qq.com', '47532486382', '12345');",
            "INSERT INTO student VALUES ('10002', 'Rose', 'female', '10002@qq.com', '83746843854', '12345');",
            "INSERT INTO student VALUES ('10003', 'Michael', 'male', '10003@qq.com', '297865834536', '12345');",
            "INSERT INTO student VALUES ('10004', 'Hepburn', 'female', '10004@qq.com', '62939849542', '12345');",
            "INSERT INTO student VALUES ('10005', 'Lisa', 'female', '10005@qq.com', '20758468732', '12345');"]
for i in sql_insert_student:
    cursor.execute(i)

sql_insert_teacher = ["INSERT INTO teacher VALUES ('20001', 'Geller', 'male', '20001@qq.com', 'Computer_Science,Data_Structure', '45678');",
            "INSERT INTO teacher VALUES ('20002', 'Bing', 'male', '20002@qq.com', 'Advanced_Math, Linear_Algebra', '45678');",
            "INSERT INTO teacher VALUES ('20003', 'Tyrell', 'male', '20003@qq.com', 'Operating_System, C++', '45678');",
            "INSERT INTO teacher VALUES ('20004', 'Green', 'female', '20004@qq.com', 'Python, Java', '45678');",
            "INSERT INTO teacher VALUES ('20005', 'Joey', 'male', '20005@qq.com', 'Database', '45678');"]
for i in sql_insert_teacher:
    cursor.execute(i)

sql_insert_course = ["INSERT INTO course VALUES ('30001', 'Computer_Science', 'Geller');",
            "INSERT INTO course VALUES ('30002', 'Data_Structure', 'Geller');",
            "INSERT INTO course VALUES ('30003', 'Advanced_Math', 'Bing');",
            "INSERT INTO course VALUES ('30004', 'Linear_Algebra', 'Bing');",
            "INSERT INTO course VALUES ('30005', 'Operating_System', 'Tyrell');",
            "INSERT INTO course VALUES ('30006', 'C++', 'Tyrell');",
            "INSERT INTO course VALUES ('30007', 'Python', 'Green');",
            "INSERT INTO course VALUES ('30008', 'Java', 'Green');",
            "INSERT INTO course VALUES ('30009', 'Database', 'Joey');"]
for i in sql_insert_course:
    cursor.execute(i)

connect.commit()
connect.close()