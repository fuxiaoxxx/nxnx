import pymysql.cursors

# 展示学生信息
def show_stu_info(sno, cursor):
    sql = "select Sno, Sname, Ssex, E_mail, Phone_num from student where sno = %s;"
    cursor.execute(sql % sno)
    data = cursor.fetchall()
    print("Sno: %s, Sname: %s, Ssex: %s \nE_mail: %s \nPhone_num: %s\n" % data[0])

# 修改学生信息
def updata_info(sno, col, new_data, cursor):
    sql = "update student set %s = '%s' where Sno = %s;"
    try:
        cursor.execute(sql % (col, new_data, sno))
        connect.commit()
        print("成功修改")
        show_stu_info(sno, cursor)
    except:
        print("修改失败，请重试")

#修改密码
def update_password(sno, new_psw, cursor):
    cursor.execute("update student set %s = '%s' where Sno = %s;" % ("Password", new_psw, sno))
    connect.commit()
    print("密码修改完成")

# 展示所有课程信息
def show_course(cursor):
    sql = "select * from course;"
    cursor.execute(sql)
    for item in cursor.fetchall():
        print("Course ID: %s, name: %s, teacher: %s" % item)

# 展示所选课程成绩
def show_selected_score(sno, cursor):
    print("所有课程成绩信息如下:")
    sql = "select CID, Cname, teacher, score from student_score where Sno = %s;"
    cursor.execute(sql % sno)
    data1 = cursor.fetchall()

    for item in data1:
        CID = item[0]
        tch = item[2]
        sc = item[3]
        if sc == None:
            sc = 0
        cursor.execute("select count(distinct Sno) from student_score where CID = '%s' and teacher = '%s' and score > %d;" % (CID, tch, sc))
        data2 = cursor.fetchall()
        lst = list(item)
        lst.append(data2[0][0] + 1)
        print("课程ID: %s, 课程: %s, 讲师: %s, 分数: %s, 排名: %d" % tuple(lst))

# 展示所选课程信息
def show_selected_course(Sno, cursor):
    sql = "select CID, Cname, teacher from pick_up where Sno = %s;"
    cursor.execute(sql % Sno)

    selected_courses = cursor.fetchall()

    for i in selected_courses:
        print("Course ID: %s, Course name: %s, Teacher: %s" % i)
    return selected_courses

# 选课
def pick_up_course(Sno, Sna, CID, cursor):
    cursor.execute("select * from pick_up where Sno = '%s' and CID = '%s';" % (Sno, CID))
    data = cursor.fetchall()
    if data != tuple():
        print("该学生已选上该课程，无需再选")
    else:
        cursor.execute("select Cname, Cteacher, CID from course where CID = %s" % CID)
        nt = cursor.fetchall()
        if nt == tuple():
            print("无此课程")
        else:
            print(nt)
            cn, ct, ci= nt[0][0], nt[0][1], nt[0][2]

            cursor.execute("INSERT INTO pick_up VALUES ('%s', '%s', '%s', '%s', '%s');" % (Sna, Sno, ci, cn, ct))
            cursor.execute("INSERT INTO student_score VALUES ('%s', '%s', '%s', '%s', '%s', null);" % (Sna, Sno, ci, cn, ct))
            connect.commit()
            print("成功选上该课程,您所选择的所有课程如下:")
            show_selected_course(Sno, cursor)

# 退课
def delete_course(Sno, CID, cursor):
    selected_courses = show_selected_course(Sno, cursor)
    sql = "DELETE FROM pick_up where CID = %s AND Sno = %s"
    for data in selected_courses:
        if data[0] == CID:
            cursor.execute(sql % (CID, Sno))
            cursor.execute("DELETE FROM student_score where CID = %s AND Sno = %s" % (CID, Sno))
            connect.commit()
            print("退课后您的所有课程信息如下:")
            show_selected_course(Sno, cursor)
            break
    else:
        print("退课失败, 您未选该课程")

connect = pymysql.Connect(
    host = 'localhost',
    port = 3306,
    user = 'root',
    passwd = 'zgt228228',
    db = 'school',
    charset = 'utf8'
)

cursor = connect.cursor()
sql_ = '''
select Sno, Sname, Password from student;
'''
sno_pswd = {}
sno_sna = {}
cursor.execute(sql_)
for row in cursor.fetchall():
    sno_pswd[row[0]] = row[2]
    sno_sna[row[0]] = row[1]

sno = ""
sna = ""
psw = ""
while True:
    sno = str(input("请输入学号:"))
    psw = str(input("请输入密码:"))
    # psw = getpass.getpass(prompt="Please input your password")
    # psw = maskpass.askpass(mask="*")
    if sno in sno_sna.keys():
        if psw == sno_pswd[sno]:
            sna = sno_sna[sno]
            print("登陆成功, %s\n" % sno_sna[sno])
            show_stu_info(sno, cursor)
            break
        else:
            print("账号或密码错误")
    else:
        print("无此学生")

while True:
    i = int(input("请输入您的操作  1 表示修改邮箱  2 表示修改电话  3 表示修改密码  4 表示选课退课  5 表示查成绩  6 查看个人信息  7 表示退出\n"))
    if i == 1:
        new_data = str(input("请输入新的邮箱地址："))
        updata_info(sno, "E_mail", new_data, cursor)
    elif i == 2:
        new_data = str(input("请输入新的电话:"))
        updata_info(sno, "Phone_num", new_data, cursor)
    elif i == 3:
        new_psw = str(input("请输入新的密码:"))
        update_password(sno, new_psw, cursor)
    elif i == 4:
        print("所有课程如下所示")
        show_course(cursor)
        print("您已选择下面的课程:")
        show_selected_course(sno,cursor)
        ch = input("您要选课还是退课, a表示选课, b表示退课: ")
        if ch == 'a':
            CID = input("请输入课程ID: ")
            pick_up_course(sno, sna, CID, cursor)

        elif ch == 'b':
            CID = input("请输入要退课程的ID: ")
            delete_course(sno, CID, cursor)
        else:
            pass
    elif i == 5:
        show_selected_score(sno, cursor)

    elif i == 6:
        print("当前个人信息如下:")
        show_stu_info(sno, cursor)

    else:
        print("已退出系统, 欢迎下次使用")
        break

connect.close()