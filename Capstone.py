import copy
import dbms

def checkAccount(username):
    import mysql.connector

    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Sb@405984",
        database="employee"
    )

    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM account where username = '{}'".format(username))

    myresult = mycursor.fetchall()

    if myresult:
        return True
    else:
        return False


def checkPassword(username, password):
    import mysql.connector
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Sb@405984",
        database="employee"
    )
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM account where username = '{}' and password = '{}'".format(username, password))

    myresult = mycursor.fetchall()

    if myresult:
        return True
    else:
        return False


def createAccount(username, password):
    import mysql.connector
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Sb@405984",
        database="employee"
    )

    mycursor = mydb.cursor()

    sql = "INSERT INTO account (username, password) VALUES (%s, %s)"
    val = (username, password)

    mycursor.execute(sql, val)

    mydb.commit()
    print(mycursor.rowcount, "record inserted.")





class Matrix:

    def __init__(self, inper):
        self.matrix = None
        self.rows, self.columns = inper.split()

    def create(self):
        self.matrix = [[float(n) for n in input().split()] for _row in range(int(self.rows))]

    def add(self, mtx):
        if self.rows == mtx.rows and self.columns == mtx.columns:
            result = [[self.matrix[i][j] + mtx.matrix[i][j] for j in range(len(self.matrix[0]))] for i in
                      range(len(self.matrix))]
            for row in result:
                for number in row:
                    print(number, end=" ")
                print()
        else:
            print("ERROR")

    def subtract(self, mtx):
        if self.rows == mtx.rows and self.columns == mtx.columns:
            result = [[self.matrix[i][j] - mtx.matrix[i][j] for j in range(len(self.matrix[0]))] for i in
                      range(len(self.matrix))]
            for row in result:
                for number in row:
                    print(number, end=" ")
                print()
        else:
            print("ERROR")

    def multiply(self, number):
        result = [[self.matrix[i][j] * number for j in range(len(self.matrix[0]))] for i in range(len(self.matrix))]
        for row in result:
            for number in row:
                print(round(number, 4), end=" ")
            print()

    def transpose(self):
        transposed = [[self.matrix[i][j] for i in range(len(self.matrix))] for j in range(len(self.matrix[0]))]
        self.matrix = transposed

    def transpose_side(self):
        transposed = [[self.matrix[i][j] for i in range(-1, -len(self.matrix) - 1, -1)] for j in
                      range(-1, -len(self.matrix[0]) - 1, -1)]
        self.matrix = transposed

    def transpose_vertical(self):
        for row in self.matrix:
            row.reverse()

    def transpose_horizontal(self):
        self.matrix.reverse()

    def printer(self):
        for row in self.matrix:
            for number in row:
                print(int(number), end=" ")
            print()

    def multiply_matrices(self, mtx):
        if self.columns != mtx.rows:
            print("The operation cannot be performed.")
        else:
            result = [[sum([self.matrix[i][k] * mtx.matrix[j][k] for k in range(len(mtx.matrix[0]))]) for j in
                       range(len(mtx.matrix))] for i in range(len(self.matrix))]

            for row in result:
                for number in row:
                    print(round(number, 2), end=" ")
                print()

    @staticmethod
    def determinant(mtx):
        if len(mtx) == 1:
            return mtx[0][0]
        elif len(mtx) == 2:
            det = mtx[0][0] * mtx[1][1] - mtx[1][0] * mtx[0][1]
            return det
        else:
            recur = 0
            for i, e in enumerate(mtx):
                rex = mtx[0][i] * Matrix.determinant(
                    [[el for ind, el in enumerate(matx) if ind != i] for matx in mtx[1:]])
                if i % 2 == 0:
                    recur += rex
                else:
                    recur -= rex
            return recur

    @staticmethod
    def create_identity_matrix(siz):
        size = int(siz.split()[0])
        return [[1 if i == j else 0 for i in range(size)] for j in range(size)]

    @staticmethod
    def cofactor_matrix(size, mtx):
        cofa = []
        for i in range(len(mtx)):
            temp_cof = []
            for j in range(len(mtx[0])):
                temp_mtx = copy.deepcopy(mtx)
                temp_mtx.pop(i)
                # print(temp_mtx)
                for mitx in temp_mtx:
                    # print(mitx)
                    mitx.pop(j)
                    # print(mitx)
                cof_el = Matrix.determinant(temp_mtx) * (-1) ** (i + j)
                temp_cof.append(cof_el)
            cofa.append(temp_cof)
        c = Matrix(size)
        c.matrix = cofa
        return c


def LUdecompose():
    from numpy import double

    n = int(input("Enter the order of the matrix: "))
    # Creating the dummy Matrix table
    for x in range(n):
        print("|", end="  ")
        for y in range(n):
            print("a{}{}".format(x + 1, y + 1), end="  ")
        print("|", end="")
        print("\n")

    print("For this representation enter the values respectively: ")
    A = []

    for x in range(n):
        A.append([0])
        for y in range(n):
            A[x][y] = int(input("a{}{}: ".format(x + 1, y + 1)))
            A[x].append(0)
        A[x].pop()

    print(A)

    L = []
    for x in range(n):
        L.append([0])
        for y in range(n):
            if x == y:
                L[x][y] = 1
            else:
                L[x][y] = 0
            L[x].append(0)
        L[x].pop()

    k = copy.deepcopy(n - 1)
    U = copy.deepcopy(A)
    while k > 0:
        for x in range(n - k, n):
            if U[x][n - k - 1] != 0:
                temp = -U[x][n - k - 1] / U[n - k - 1][n - k - 1]
                U[x][n - k - 1] = -temp
                for y in range(n - k - 1, n):
                    U[x][y] = U[x][y] + (temp * U[n - k - 1][y])
        k = k - 1
    print(L)
    print(U)


def Trace():
    def findTrace(mat, n):
        sum = 0
        for i in range(n):
            sum += mat[i][i]
        return sum

    n = int(input("Enter the order of the matrix: "))
    # Creating the dummy Matrix table
    for x in range(n):
        print("|", end="  ")
        for y in range(n):
            print("a{}{}".format(x + 1, y + 1), end="  ")
        print("|", end="")
        print("\n")

    print("For this representation enter the values respectively: ")
    A = []

    for x in range(n):
        A.append([0])
        for y in range(n):
            A[x][y] = int(input("a{}{}: ".format(x + 1, y + 1)))
            A[x].append(0)
        A[x].pop()

    print("Trace of Matrix =", findTrace(A, n))


def Scalor():
    a = []
    for i in range(3):
        b = []
        for j in range(3):
            j = int(input("enter the elements in 1st matrix : [" + str(i) + "][" + str(j) + "]"))
            b.append(j)
        a.append(b)

    print("the matrix is:")
    for i in range(3):
        for j in range(3):
            print(a[i][j], end=" ")
        print()

    f = 0
    diag = a[0][0]
    for i in range(3):
        for j in range(3):
            if i == j and a[i][j] != diag:
                f = 1
                break

    if f == 0:
        print("it's is a scalar matrix")
    else:
        print("it's not a scalar matrix")


def Identical():
    def equal(a, c):

        for i in range(3):

            for j in range(3):

                if a[i][j] != c[i][j]:
                    return 0

        return 1

    a = []

    for i in range(3):

        b = []

        for j in range(3):
            j = int(input("enter the elements in 1st matrix : [" + str(i) + "][" + str(j) + "]"))

            b.append(j)

        a.append(b)

    c = []

    for i in range(3):

        d = []

        for j in range(3):
            j = int(input("enter the elements in 2nd matrix : [" + str(i) + "][" + str(j) + "]"))

            d.append(j)

        c.append(d)

    print("first matrix is:")

    for i in range(3):

        for j in range(3):
            print(a[i][j], end=" ")

        print()

    print("second matrix is:")

    for i in range(3):
        for j in range(3):
            print(c[i][j], end=" ")

        print()

    if equal(a, c) == 0:

        print("matrices are not equal")

    else:

        print("matrices are equal")


def Power():
    from numpy import linalg as LA
    import numpy as np
    choice = "y"
    while (choice == "y") or (choice == "Y"):
        R = int(input("Enter the no of rows: "))
        C = int(input("Enter the no of columns: "))

        if R == C:
            print("Enter the entries in a single line (separated by spaces): ")

            entries = list(map(int, input().split()))
            n = int(input("Enter the power: "))

            matrix = np.array(entries).reshape(R, C)
            print(matrix)
            print("Power : ", n)
            print("\n")
            print("Matrix Power is: \n", LA.matrix_power(matrix, n))

        else:
            print("\n")
            print("...-- ERROR --...")
            print("Number of columns and rows should be equal.")

        print("\n")
        choice = input("Do you want to continue ? (Y/N) : ")


def Division():
    import numpy as np

    rows = int(input("Enter the Number of rows : "))
    column = int(input("Enter the Number of Columns: "))

    print("Enter the elements of Matrix:")
    matrix = [[int(input()) for i in range(column)] for i in range(rows)]
    print("-------Your  Matrix is---------")
    for n in matrix:
        print(n)

    divisor = int(input("Enter the divisor : "))
    print("martix is   : ", matrix)

    # output_array
    out = np.divide(matrix, divisor)
    print("\nOutput array : \n", out)


def Diagonalization():
    import numpy as np
    choice = "y"
    while (choice == "y") or (choice == "Y"):
        R = int(input("Enter the no of rows: "))
        C = int(input("Enter the no of columns: "))

        if (R == C):
            print("Enter the enteries in a single line (separated by spaces): ")

            entries = list(map(int, input().split()))

            matrix = np.array(entries).reshape(R, C)
            print(matrix)
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            P = eigenvectors

            D = np.zeros((R, C))
            for i in range(R):
                D[i, i] = eigenvalues[i]

            P_inv = np.linalg.inv(P)

            X = np.dot(P_inv, matrix)
            B = np.dot(X, P)
            print("\n")
            print(D)

        else:
            print("\n")
            print("The number of columns and rows should be equal.")

        print("\n")
        choice = input("Do you want to continue ? (Y/N) : ")


def Cramers():
    a = float(input("Enter a: "))
    b = float(input("Enter b: "))
    c = float(input("Enter c: "))
    d = float(input("Enter d: "))
    e = float(input("Enter e: "))
    f = float(input("Enter f: "))

    ##a*x + by = e
    ##cx + dy = f

    if (a * d - b * c == 0):
        print("The equation has no solution")
    else:
        x = (e * d - b * f) / (a * d - b * c)
        y = (a * f - e * d) / (a * d - b * c)

        print("x=%s" % x, "y=%s" % y)


def Eigen():
    import numpy as np

    mat = np.mat("1 -2;1 3")

    # Original matrix
    print(mat)
    print("")
    evalue, evect = np.linalg.eig(mat)

    # Eigenvalues
    print(evalue)
    print("")

    # Eigenvectors
    print(evect)


def menu(username):
    while True:
        print("HELLO {}".format(username))
        print("1. Add matrices\n2. subtract matrices\n3. Multiply matrix by a constant\n4. Multiply matrices\n\
5. Transpose matrix\n6. Calculate a determinant\n7. Inverse matrix \n8. LUDecomposition\n9. Trace \n10. Scalor\n11. Identical Matrices \n12. Power of Matrix \n\
13. Division of matrix \n14. Diagonalization of matrix \n15. Cramer's rule \n16. Eigen Vectors and Values \n17. Create new Account\n0. Exit")
        choice = input()
        if choice == "1":
            print("\033[H\033[2J");
            print("ADDITION OF MATRICES")
            print("Enter size(row&column) of first matrix: ")
            matrix_a = Matrix(input())
            print("Enter first matrix: ")
            matrix_a.create()

            print("Enter size(row&column) of second matrix: ")
            matrix_b = Matrix(input())
            print("Enter second matrix: ")
            matrix_b.create()

            print("The result is:")
            matrix_a.add(matrix_b)
            input("Press Enter to continue...")

        elif choice == "2":
            print("\033[H\033[2J");
            print("SUBTRACTION OF MATRICES")
            print("Enter size(row&column) of first matrix: ")
            matrix_a = Matrix(input())
            print("Enter first matrix: ")
            matrix_a.create()

            print("Enter size(row&column) of second matrix: ")
            matrix_b = Matrix(input())
            print("Enter second matrix: ")
            matrix_b.create()

            print("The result is:")
            matrix_a.subtract(matrix_b)
            input("Press Enter to continue...")


        elif choice == "3":
            print("\033[H\033[2J");
            print("MULTIPLICATION OF MATRICES USING A CONSTANT")
            print("Enter size of matrix: ")
            matrix_a = Matrix(input())
            print("Enter matrix: ")
            matrix_a.create()

            const = int(input("Enter constant: "))
            print("The result is:")
            matrix_a.multiply(const)
            input("Press Enter to continue...")


        elif choice == "4":
            print("\033[H\033[2J");
            print("MULTIPLICATION OF MATRICES")
            print("Enter size of first matrix: ")
            matrix_a = Matrix(input())
            print("Enter first matrix: ")
            matrix_a.create()

            print("Enter size of second matrix: ")
            matrix_b = Matrix(input())
            print("Enter second matrix: ")
            matrix_b.create()
            matrix_b.transpose()
            # print(matrix_b.matrix)
            print("The result is:")
            matrix_a.multiply_matrices(matrix_b)
            input("Press Enter to continue...")


        elif choice == "5":
            print("\033[H\033[2J");
            print("TRANSPOSE OF A MATRIX")
            print('1. Main diagonal\n2. Side diagonal\n3. Vertical line\n4. Horizontal line')
            tran_choice = input("Your choice: ")
            print("Enter size of matrix: ")
            matrix_a = Matrix(input())
            print("Enter matrix: ")
            matrix_a.create()
            if tran_choice == "1":
                matrix_a.transpose()
            elif tran_choice == "2":
                matrix_a.transpose_side()
            elif tran_choice == "3":
                matrix_a.transpose_vertical()
            elif tran_choice == "4":
                matrix_a.transpose_horizontal()

            matrix_a.printer()
            input("Press Enter to continue...")


        elif choice == "6":
            print("\033[H\033[2J");
            print("DETERMINANT OF A MATRIX")
            print("Enter size of matrix: ")
            matrix_a = Matrix(input())
            print("Enter matrix: ")
            matrix_a.create()
            print("The result is:")
            print(matrix_a.determinant(matrix_a.matrix))
            input("Press Enter to continue...")


        elif choice == "7":
            print("\033[H\033[2J");
            print("INVERSE OF A MATRIX")
            size = input("Enter size of matrix: ")
            matrix_a = Matrix(size)
            print("Enter matrix: ")
            matrix_a.create()
            det_a = matrix_a.determinant(matrix_a.matrix)

            iden = Matrix.create_identity_matrix(size)
            matrix_c = Matrix.cofactor_matrix(size, matrix_a.matrix)
            matrix_c.transpose()
            # print(matrix_c.matrix)
            # print(det_a)
            if det_a:
                print("The result is:")
                matrix_c.multiply(1 / det_a)
            else:
                print("This matrix doesn't have an inverse.")
                input("Press Enter to continue...")

        elif choice == "8":
            print("\033[H\033[2J");
            print("LU DECOMPOSITION OF A MATRIX")
            LUdecompose()
            input("Press Enter to continue...")

        elif choice == "9":
            print("\033[H\033[2J");
            print("TRACE OF A MATRIX")
            Trace()
            input("Press Enter to continue...")

        elif choice == "10":
            print("\033[H\033[2J");
            print("CHECK IF A MATRIX IS SCALOR OR NOT")
            Scalor()
            input("Press Enter to continue...")

        elif choice == "11":
            print("\033[H\033[2J");
            print("CHECK IF TWO MATRIXES ARE IDENTICAL")
            Identical()
            input("Press Enter to continue...")

        elif choice == "12":
            print("\033[H\033[2J");
            print("POWER OF A MATRIX")
            Power()
            input("Press Enter to continue...")

        elif choice == "13":
            print("\033[H\033[2J");
            print("DIVISION OF A MATRIX USING A CONSTANT")
            Division()
            input("Press Enter to continue...")

        elif choice == "14":
            print("\033[H\033[2J");
            print("DIAGONALIZATION OF A MATRIX")
            Diagonalization()
            input("Press Enter to continue...")

        elif choice == "15":
            print("\033[H\033[2J");
            print("SOLVING TWO LINEAR EQUATION OF TWO VARIABLE USING CRAMER'S RULE")
            Cramers()
            input("Press Enter to continue...")

        elif choice == "16":
            print("\033[H\033[2J");
            print("EIGEN VALUE AND EIGEN VECTOR OF A MATRIX")
            Eigen()
            input("Press Enter to continue...")

        elif choice == "17":
            username = str(input("Enter your username: "))
            password = str(input("Enter your password: "))
            print(createAccount(username, password))
            input("Press Enter to continue...")
            print("\033[H\033[2J");

        elif choice == "0":
            break
            input("Press Enter to continue...")

        else:
            input("Enter correct choice!!..")

print("\033[H\033[2J");
print("HELLO EVERYONE!")
print("This is team 4 going to present the python Capstone Project on the topic MATRIX CALCULATIONS")
print("Team members are: \n1. Sayan\n2. Eliyajer\n3. Manikanta\n4. Minaal\n5. Sanjay")

while True:
    print("Log In: ")
    username = str(input("Enter your username: "))
    if checkAccount(username):
        password = str(input("Enter your password: "))
        if checkPassword(username, password):
            print("\033[H\033[2J");
            print("WELCOME {}".format(username))
            menu(username)

        else:
            print("Incorrect Password!")
    else:
        print("Account doesnot Exist!!")