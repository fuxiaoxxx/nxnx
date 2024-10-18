import pymysql.cursors

# 展示教师信息
def show_tch_info(Tno, cursor):
    sql = "select Tno, Tname, Tsex, E_mail, courses from teacher where Tno = %s"
    cursor.execute(sql % Tno)
    data = cursor.fetchall()
    print("学工号: %s, 姓名: %s, 性别: %s \nE_mail: %s \n所教课程: %s\n" % data[0])

# 修改E_mail
def updata_e_mail(Tno, new_data, cursor):
    sql = "update teacher set E_mail = '%s' where Tno = %s;"

    try:
        cursor.execute(sql % (new_data, Tno))
        connect.commit()
        print("修改成功")
        show_tch_info(Tno, cursor)
    except:
        print("修改失败，请重试")

# 修改密码
def updata_password(Tno, new_data, cursor):
    try:
        cursor.execute("update teacher set Password = '%s' where Tno = %s;" % (new_data, Tno))
        connect.commit()
        print("密码修改完成")
    except:
        print("修改失败，请重试")

# 展示所有学生信息
def show_stu_info(cursor):
    sql = "select Sno, Sname, Ssex, E_mail, Phone_num from student;"
    cursor.execute(sql)
    print("所有学生信息如下:")
    for data in cursor.fetchall():
        print("学号: %s  姓名: %s  性别: %s  电子邮件: %s  手机号: %s" % (data))

# 增加一个课程
def add_course(CID, Cname, Tno, cursor):
    cursor.execute("select courses, Tname from teacher where Tno = %s;" % Tno)
    data = cursor.fetchall()
    courses = data[0][0]
    courses += ","
    courses += Cname

    cursor.execute("update teacher set courses = '%s' where Tno = %s;" % (courses, Tno))
    cursor.execute("INSERT INTO course VALUES ('%s', '%s', '%s');" % (CID, Cname, data[0][1]))
    connect.commit()
    print("成功增加一个课程")

# 减少一个课程
def delete_course(CID, Tno, Tna, cursor):
    cursor.execute("select Cname, Cteacher from course where CID = '%s' AND Cteacher = '%s';" % (CID, Tna))
    data = cursor.fetchall()
    if data == tuple():
        print("输入有误，您不教授该课程")
    else:
        cursor.execute("select courses, Tname from teacher where Tno = %s;" % Tno)
        data2 = cursor.fetchall()
        courses = str(data2[0][0])
        if courses.__contains__("," + data[0][0]):
            courses = courses.replace("," + data[0][0], "")
        elif courses.__contains__(data[0][0] + ","):
            courses = courses.replace(data[0][0] + "," , "")
        else :
            courses = courses.replace(data[0][0], "")
        cursor.execute("update teacher set courses = '%s' where Tno = %s;" % (courses, Tno))

        cursor.execute("DELETE FROM course WHERE Cteacher = '%s' AND CID = '%s';" % (Tna, CID))
        cursor.execute("DELETE FROM pick_up WHERE teacher = '%s' AND CID = '%s';" % (Tna, CID))
        cursor.execute("DELETE FROM student_score WHERE teacher = '%s' AND CID = '%s';" % (Tna, CID))
        connect.commit()
        print("成功删除该课程")

# 展示选课学生的成绩信息
def show_stu_score(Tna, cursor):
    print("选课学生成绩如下所示：")
    cursor.execute("select CID from course where Cteacher = '%s'" % Tna)
    course_id = cursor.fetchall()
    for id_ in course_id:
        id = id_[0]
        cursor.execute("select Sname, Sno, CID, Cname, score from student_score where teacher = '%s' AND CID = '%s' order by score desc;" % (Tna, id))
        for item in cursor.fetchall():
            print("姓名: %s  学号: %s  课程ID: %s  课程名: %s  分数: %s " % item)

# 修改某学生的成绩
def updata_stu_score(Sno, CID, Tna, new_socre, cursor):
    cursor.execute("select * from student_score where Sno = '%s' and CID = '%s' and teacher = '%s'" % (Sno, CID, Tna))
    if cursor.fetchall() == tuple():
        print("错误的学生信息或课程信息，请重试")
    else:
        try:
            cursor.execute("update student_score set score = %d where Sno = '%s' and CID = '%s';" % (new_socre, Sno, CID))
            print("修改成功")
            connect.commit()
        except:
            print("修改失败请重试")



connect = pymysql.Connect(
    host = 'localhost',
    port = 3306,
    user = 'root',
    passwd = '123456',
    db = 'school',
    charset = 'utf8'
)
cursor = connect.cursor()

sql_ = '''
select Tno, Tname, Password from teacher;
'''
tno_pswd = {}
tno_tna = {}
cursor.execute(sql_)
for row in cursor.fetchall():
    tno_pswd[row[0]] = row[2]
    tno_tna[row[0]] = row[1]

tno = ""
tna = ""
psw = ""
while True:
    tno = str(input("请输入工号:"))
    psw = str(input("请输入密码:"))
    # psw = getpass.getpass(prompt="Please input your password")
    # psw = maskpass.askpass(mask="*")
    if tno in tno_tna.keys():
        if psw == tno_pswd[tno]:
            tna = tno_tna[tno]
            print("登陆成功, %s\n" % tno_tna[tno])
            show_tch_info(tno, cursor)
            break
        else:
            print("账号或密码错误")
    else:
        print("无此教师")

while True:
    i = int(input("请输入您的操作  1 修改邮箱  2 修改密码  3 查看所有学生信息  4 查看选课学生成绩  5 修改学生成绩  6 修改教授课程  7 表示退出\n"))
    if i == 1:
        new_data = str(input("请输入新的邮箱地址："))
        updata_e_mail(tno, new_data, cursor)
    elif i == 2:
        new_data = str(input("请输入新的密码:"))
        updata_password(tno, new_data, cursor)
    elif i == 3:
        show_stu_info(cursor)
    elif i == 4:
        show_stu_score(tna, cursor)
    elif i == 5:
        print("所有选课学成绩如下：")
        show_stu_score(tna, cursor)
        sno = input("请输入该生学号: ")
        cid = input("请输入课程的ID: ")
        new_socre = int(input("请输入修改后的成绩: "))
        updata_stu_score(sno, cid, tna, new_socre, cursor)
    elif i == 6:
        print("您教授课程如下所示")
        cursor.execute("select CID, Cname from course where Cteacher = '%s';" % tna)
        print(cursor.fetchall())

        ch = input("您要增加一门课程还是减少一门课程, a表示增加, b表示减少: ")
        if ch == 'a':
            CID = input("请输入课程ID: ")
            Cname = input("请输入课程名称: ")
            add_course(CID, Cname, tno, cursor)

        elif ch == 'b':
            CID = input("请输入要退课程的ID: ")
            delete_course(CID, tno, tna, cursor)
        else:
            pass

    else:
        print("已退出系统, 欢迎下次使用")
        break

connect.close()