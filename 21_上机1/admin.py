import pymysql.cursors

# 展示所有学生信息
def show_stu_info(cursor):
    cursor.execute("select * from student;")
    for data in cursor.fetchall():
        print("学号: %s  姓名: %s  性别: %s  E_mail: %s  电话: %s  密码: %s" % data)

# 展示所有教师信息
def show_tch_info(cursor):
    cursor.execute("select * from teacher;")
    for data in cursor.fetchall():
        print("工号: %s  姓名: %s  性别: %s  E_mail: %s  教授课程: %s  密码: %s" % data)

# 展示所有课程信息
def show_cs_info(cursor):
    cursor.execute("select * from course;")
    for data in cursor.fetchall():
        print("课程ID: %s  课程: %s  讲师: %s" % data)

# 添加学生
def insert_stu(stu_info, cursor):
    cursor.execute("INSERT INTO student VALUES ('%s','%s','%s','%s','%s','%s')" % stu_info)
    connect.commit()
    print("添加成功")

# 添加教师
def insert_tch(tch_info, cursor):
    cursor.execute("INSERT INTO teacher VALUES ('%s','%s','%s','%s','%s','%s')" % tch_info)
    connect.commit()
    print("添加成功")

# 添加课程
def insert_cs(cs_info, cursor):
    cursor.execute("select CID, Cname, Cteacher from course where CID = '%s'" % cs_info[0])
    data = cursor.fetchall()
    if cs_info in data:
        print("该课程信息已存在，请检查后重新添加")
    elif data != tuple() and data[0][1] != cs_info[1]:
        print("课程ID已经存在，请检查后重新添加")
    else:
        cursor.execute("select courses from teacher where Tname = '%s';" % cs_info[2])
        data2 = cursor.fetchall()
        if data2 == tuple():
            pass
        else:
            cs = str(data2[0][0])
            if cs.__contains__(cs_info[1]):
                pass
            else:
                cs += ","
                cs += cs_info[1]
                cursor.execute("update teacher set courses = '%s' where Tname = '%s';" % (cs, cs_info[2]))
                connect.commit()


        cursor.execute("INSERT INTO course VALUES ('%s','%s','%s')" % cs_info)
        connect.commit()
        print("添加成功")

# 删除学生
def delete_stu(Sno, cursor):
    cursor.execute("DELETE FROM student where Sno = '%s'" % Sno)
    cursor.execute("DELETE FROM pick_up where Sno = '%s'" % Sno)
    cursor.execute("DELETE FROM student_score where Sno = '%s'" % Sno)
    connect.commit()
    print("删除成功")

# 删除教师
def delete_tch(Tno, cursor):
    cursor.execute("SELECT Tname, courses FROM teacher WHERE Tno = '%s';" % Tno)
    data = cursor.fetchall()
    if data == tuple():
        print("错误的工号，请检查后重新输入")
    else:
        Tna = data[0][0]
        courses = str(tuple(str(data[0][1]).split(",")))
        if len(tuple(str(data[0][1]).split(","))) == 1:
            courses = courses.replace(",", "")
        cursor.execute("DELETE FROM teacher where Tno = '%s';" % Tno)
        cursor.execute("DELETE FROM course where Cteacher = '%s' AND Cname IN %s;" % (Tna, str(courses)))
        cursor.execute("DELETE FROM pick_up where teacher = '%s' AND Cname IN %s;" % (Tna, str(courses)))
        cursor.execute("DELETE FROM student_score where teacher = '%s' AND Cname IN %s;" % (Tna, str(courses)))

        connect.commit()

# 删除课程
def delete_cs(CID, Tna, cursor):
    cursor.execute("select Cname from course where CID = '%s' and Cteacher = '%s';" % (CID, Tna))
    cn = cursor.fetchall()
    if cn == tuple():
        print("输入信息有误，请检查后重新输入")
    else:
        # 在教师表中删除教师的该课程
        cursor.execute("select courses from teacher where Tname = '%s';" % Tna)
        data2 = cursor.fetchall()
        if data2 == tuple():
            pass
        else:
            courses = str(data2[0][0])
            if courses.__contains__("," + cn[0][0]):
                courses = courses.replace("," + cn[0][0], "")
            elif courses.__contains__(cn[0][0] + ","):
                courses = courses.replace(cn[0][0] + ",", "")
            else:
                courses = courses.replace(cn[0][0], "")
            cursor.execute("update teacher set courses = '%s' where Tname = '%s';" % (courses, Tna))

        cursor.execute("DELETE FROM course WHERE Cteacher = '%s' AND CID = '%s';" % (Tna, CID))
        cursor.execute("DELETE FROM pick_up WHERE teacher = '%s' AND CID = '%s';" % (Tna, CID))
        cursor.execute("DELETE FROM student_score WHERE teacher = '%s' AND CID = '%s';" % (Tna, CID))
        connect.commit()
        print("成功删除该课程")

# 修改学生教师邮箱电话和密码
def update_stu_tch_info(table, no, col, new_data, cursor):
    cursor.execute("select * from %s where %sno = '%s';" % (table, str(table)[0].upper(), no))
    data = cursor.fetchall()
    if data == tuple():
        print("错误信息，请检查后重试")
    else:
        cursor.execute("update %s set %s = '%s' where %sno = '%s';" % (table, col, new_data, str(table)[0].upper(), no))
        print("修改成功")
        connect.commit()

# 修改学生选课信息
def update_pick_up_info(Sno, CID, cursor):
    cursor.execute("select * from pick_up where Sno = '%s' and CID = '%s';" % (Sno, CID))
    data = cursor.fetchall()
    if data != tuple():
        print("该学生已选上该课程，无需再选")
    else:

        cursor.execute("select Sname from student where Sno = '%s';" % Sno)
        data2 = cursor.fetchall()
        if data2 == tuple():
            print("学号有误，请检查后重新输入")
        else:
            Sna = data2[0][0]

            cursor.execute("select Cname, Cteacher, CID from course where CID = %s" % CID)
            nt = cursor.fetchall()
            print(nt)
            cn, ct, ci = nt[0][0], nt[0][1], nt[0][2]

            cursor.execute("INSERT INTO pick_up VALUES ('%s', '%s', '%s', '%s', '%s');" % (Sna, Sno, ci, cn, ct))
            cursor.execute("INSERT INTO student_score VALUES ('%s', '%s', '%s', '%s', '%s', null);" % (Sna, Sno, ci, cn, ct))
            connect.commit()
            print("成功帮助该同学选上该课程")

# 退课
def delete_pick_up(Sno, CID, cursor):
    sql = "DELETE FROM %s where CID = %s AND Sno = %s"
    cursor.execute(sql % ("pick_up", CID, Sno))
    cursor.execute(sql % ("student_score", CID, Sno))
    connect.commit()
    print("修改成功")

# 修改学生成绩
def update_score(Sno, CID, new_score, cursor):
    cursor.execute("select * from pick_up where Sno = '%s' and CID = '%s';" % (Sno, CID))
    data = cursor.fetchall()
    if data == tuple():
        print("输入信息有误，请检查后重新输入")
    else:
        cursor.execute("update student_score set score = %d where Sno = '%s' and CID = '%s';" % (new_score, Sno, CID))
        print("修改成功")
        connect.commit()

# 修改课程ID和名称
def updata_cs_info(old_id, col, new_data, cursor):
    if col == "Cname":
        cursor.execute("select Cname, Cteacher from course where CID = '%s';" % old_id)
        data = cursor.fetchall()
        if data == tuple():
            print("输入有误，请检查后重试")
            return
        else:
            cna = data[0][0]
            tna = data[0][1]
            cursor.execute("select courses from teacher where Tname = '%s';" % tna)
            temp = cursor.fetchall()
            if temp == tuple():
                print("出现错误。请检查后重试")
            else:
                old_cs = str(temp[0][0])
                new_cs = old_cs.replace(cna, new_data)
                cursor.execute("update teacher set courses = '%s' where Tname = '%s';" % (new_cs, tna))
                connect.commit()

    cursor.execute("select * from course where CID = '%s';" % old_id)
    if cursor.fetchall() == tuple():
        print("输入数据有误，请检查后重新输入")

    else:
        cursor.execute("update course set %s = '%s' where CID = '%s';" % (col, new_data, old_id))
        cursor.execute("update pick_up set %s = '%s' where CID = '%s';" % (col, new_data, old_id))
        cursor.execute("update student_score set %s = '%s' where CID = '%s';" % (col, new_data, old_id))
        connect.commit()
        print("修改成功")

# 查所有学生成绩表
def show_student_score_info(cursor):
    cursor.execute("select * from student_score;")
    for data in cursor.fetchall():
        print("学生: %s  学号: %s  课程ID: %s  课程: %s  授课教师: %s 分数: %s" % data)

# 查某位学生成绩
def show_score_by_stu(Sno, cursor):
    cursor.execute("select * from student_score where Sno = '%s' order by score desc;" % Sno)
    data = cursor.fetchall()
    if data == tuple():
        print("该生暂无选课信息")
        return False
    else:
        i = 1
        for item in data:
            print(i, " 学生: %s  学号: %s  课程ID: %s  课程: %s  授课教师: %s 分数: %s" % item)
            i += 1
        return True

# 查某门课程选课学生的成绩
def show_score_by_cs(CID, cursor):
    cursor.execute("select * from student_score where CID = '%s' order by score desc;" % CID)
    data = cursor.fetchall()
    if data == tuple():
        print("暂无信息，请检查后重试")
    else:
        i = 1
        for item in data:
            print(i, " 学生: %s  学号: %s  课程ID: %s  课程: %s  授课教师: %s 分数: %s" % item)
            i += 1

connect = pymysql.Connect(
    host = 'localhost',
    port = 3306,
    user = 'root',
    passwd = '123456',
    db = 'school',
    charset = 'utf8'
)
cursor = connect.cursor()

a_no = "Admin"
a_password = "admin"

while True:
    no = input("请输入管理员账号：")
    psw = input("请输入管理员密码：")
    if (no == a_no) and (psw == a_password):
        print("登陆成功")
        break
    else:
        print("账号或密码错误，请重新输入")

while True:
    i = int(input("请输入您的操作  1 查看学生信息  2 查看教师信息  3 查看课程信息  4 查看选课信息  5 增加信息  6 删除信息  7 修改信息  8 查找信息  9 表示退出\n"))
    if i == 1:
        show_stu_info(cursor)
    elif i == 2:
        show_tch_info(cursor)
    elif i == 3:
        show_cs_info(cursor)
    elif i == 4:
        show_student_score_info(cursor)

    elif i == 5:
        while True:
            ii = int(input("请输入要增加的信息: 1 学生信息  2 教师信息  3 课程信息  4 退出增加信息\n"))
            if ii == 1:
                show_stu_info(cursor)
                sno = input("请输入学生学号: ")
                sname = input("请输入学生姓名: ")
                ssex = input("请输入学生性别: ")
                e_m = input("请输入学生电子邮箱: ")
                p_n = input("请输入学生电话: ")
                psw = input("请输入学生密码: ")
                stu_info = (sno, sname, ssex, e_m, p_n, psw)
                insert_stu(stu_info, cursor)
            elif ii == 2:
                show_tch_info(cursor)
                tno = input("请输入教师工号：")
                tna = input("请输入教师姓名：")
                tsex = input("请输入教师性别：")
                e_m = input("请输入教师电子邮箱：")
                cs = input("请输入教师所教课程（多门课程中间用','隔开）：")
                psw = input("请输入教师密码：")
                tch_info = (tno, tna, tsex, e_m, cs, psw)
                insert_tch(tch_info, cursor)
            elif ii == 3:
                show_cs_info(cursor)
                cid = input("请输入课程ID：")
                cname = input("请输入课程名称：")
                tch = input("请输入授课教师：")
                cs_info = (cid, cname, tch)
                insert_cs(cs_info, cursor)
            else:
                break
    elif i == 6:
        while True:
            ii = int(input("请输入要删除的信息: 1 学生信息  2 教师信息  3 课程信息  4 退出删除信息\n"))
            if ii == 1:
                show_stu_info(cursor)
                sno = input("请输入该学生的学号：")
                delete_stu(sno, cursor)
            elif ii == 2:
                show_tch_info(cursor)
                tno = input("请输入该教师的工号：")
                delete_tch(tno, cursor)
            elif ii == 3:
                show_cs_info(cursor)
                cid = input("请输入课程ID：")
                tna = input("请输入教师姓名：")
                delete_cs(cid, tna, cursor)
            else:
                break

    elif i == 7:
        while True:
            ii = int(input("请输入要修改的信息: 1 学生邮箱  2 学生电话  3 学生密码  4 教师邮箱  5 教师密码  6 课程ID  7 课程名称  8 选课信息  9 学生成绩  10 退出修改信息\n"))
            col_list = [" ", "E_mail",  "Phone_num",  "password", "E_mail", "password", "CID", "Cname"]

            if ii <= 3:
                show_stu_info(cursor)
                no = input("请输入学生学号: ")
                new_data = input("新的数据：")
                update_stu_tch_info("student", no, col_list[ii], new_data, cursor)
            elif (ii > 3) and (ii <= 5):
                show_tch_info(cursor)
                no = input("请输入教师工号: ")
                new_data = input("新的数据：")
                update_stu_tch_info("teacher", no, col_list[ii], new_data, cursor)

            elif ii == 6:
                show_cs_info(cursor)
                old_id = input("请输入要修改课程的ID：")
                new_data = input("请输入新的课程ID：")
                updata_cs_info(old_id, "CID", new_data, cursor)
            elif ii == 7:
                show_cs_info(cursor)
                old_id = input("请输入要修改课程的ID：")
                new_data = input("请输入新的课程名称：")
                updata_cs_info(old_id, "Cname", new_data, cursor)

            elif ii == 8:
                mode = input("您是要添加选课信息还是要减少选课信息  a 添加  b 减少 ")
                if mode == "a":
                    show_stu_info(cursor)
                    sno = input("请输入学生学号：")
                    show_score_by_stu(sno, cursor)
                    show_cs_info(cursor)
                    cid = input("请输入选择课程ID：")
                    update_pick_up_info(sno, cid, cursor)
                elif mode == "b":
                    show_student_score_info(cursor)
                    sno = input("请输入学生学号：")
                    if show_score_by_stu(sno, cursor):
                        cid = input("请输入退课课程名称：")
                        delete_pick_up(sno, cid, cursor)
                else:
                    pass
            elif ii == 9:
                show_student_score_info(cursor)
                sno = input("请输入学生学号：")
                if show_score_by_stu(sno, cursor):
                    cid = input("请输入课程ID：")
                    new_score = int(input(("请输入课程新的成绩：")))
                    update_score(sno, cid, new_score, cursor)

            else:
                break

    elif i == 8:
        ii = input("请输入要查找的信息 a 查找某学生的选课及成绩  b 查看某门课程的选课人信息及成绩  c 自定义查询(输入SQL语句可实现自定义查询)  d 退出查找信息\n")
        if ii == "a":
            sno = input("请输入学生学号：")
            show_score_by_stu(sno, cursor)
        elif ii == "b":
            cid = input("请输入课程ID：")
            show_score_by_cs(cid, cursor)

        elif ii == "c":
            sql = input("请输入查找的SQL语句(要求查询字符小写):\n")
            if sql.__contains__("select"):
                cursor.execute(sql)
                for data in cursor.fetchall():
                    print(data)
            else:
                print("非合法的查询语句")

    elif i == 9:
        break


connect.close()
