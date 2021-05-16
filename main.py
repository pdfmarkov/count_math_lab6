from prettytable import PrettyTable
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def getReadyAnswer(type_answer):
    if type_answer == 1:
        print("Incorrect input.\n")
    elif type_answer == 2:
        print("No solution.\n")
    elif type_answer == 3:
        print("There is no concrete solution or it doesn't exist.\n")
    elif type_answer == 4:
        print("Convergence condition is not satisfied on this segment.\n")
    elif type_answer == 5:
        print("Counts of iteration reached 2.5 million , solution not found.\n")
    elif type_answer == 6:
        print("The initial approximation is poorly selected, solution not found.\n")
    elif type_answer == 7:
        print("Counts of iteration reached 250 thousand , solution not found.\n")


def make_graph(calculator, equation, name):
    try:
        "\t1. y' = y + (1+x)y^2\n"
        "\t2. y' = e^2x + y\n"
        "\t3. y' = y/x - 3\n"
        "\t4. y' = x^2 - 2y"
        eq_name = {1: "y' = y + (1+x)y^2",
                   2: "y' = e^2x + y",
                   3: "y' = y/x - 3",
                   4: "y' = x^2 - 2y"}
        ax = plt.gca()
        plt.grid()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        minimum = min(calculator.x_values)
        maximum = max(calculator.x_values)
        for i in calculator.dots_x:
            if i < minimum:
                minimum = i
            elif i > maximum:
                maximum = i
        x = np.linspace(minimum, maximum, 100)
        first_equation = [calculator.interpolation(i) for i in x]
        plt.title(name + ": graph of " + eq_name[equation])
        plt.plot(x, first_equation, color='r', linewidth=2)
        j = 0
        for i in calculator.x_values:
            plt.scatter(i, calculator.y_values[j], color='r', s=40)
            j += 1
        plt.show()
        del x
    except ValueError:
        return
    except ZeroDivisionError:
        return


def draw_graph(x, y, equation, name):
    try:
        "\t1. y' = y + (1+x)y^2\n"
        "\t2. y' = e^2x + y\n"
        "\t3. y' = y/x - 3\n"
        "\t4. y' = x^2 - 2y"
        eq_name = {1: "y' = y + (1+x)y^2",
                   2: "y' = e^2x + y",
                   3: "y' = y/x - 3",
                   4: "y' = x^2 - 2y"}
        ax = plt.gca()
        plt.grid()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # minimum = min(x)
        # maximum = max(x)
        # new_x = np.linspace(minimum, maximum, 100)
        plt.title(name + ": graph of " + eq_name[equation])
        # plt.plot(x, y, color='r', linewidth=2)
        for i in range(len(x)):
            plt.scatter(x[i], y[i], color='r', s=40)
        plt.show()
        del x
    except ValueError:
        return
    except ZeroDivisionError:
        return


class Interpolator:
    y_diff = []
    x_values = []
    y_values = []
    y_graph_values = []
    dots_x = []
    dots_y = []
    equation_type = 0
    step = 0

    def __init__(self, x, y, types):
        self.x_values = x
        self.y_values = y
        self.step = self.x_values[1] - self.x_values[0]
        self.y_graph_values = []
        self.y_diff = []
        self.equation_type = types
        self.dots_x = []
        self.dots_y = []

    def separated_differences(self, differences):
        if len(differences) == 1:
            return self.y_values[self.x_values.index(differences[0])]
        else:
            first = differences[0:len(differences) - 1]
            second = differences[1:len(differences)]
            return (self.separated_differences(second) - self.separated_differences(first)) / (
                    differences[-1] - differences[0])

    def interpolation(self, x):
        interpolation = self.y_diff[0][-1]
        n = len(self.x_values) - 2
        t = (x - self.x_values[-1]) / self.step
        i = 1
        while n >= 0:
            k = 1
            factor = t
            while k <= i - 1:
                factor *= (t + k) / (k + 1)
                k += 1
            interpolation += factor * self.y_diff[i][n]
            n -= 1
            i += 1
        return interpolation

    def finite_differences(self):
        i = 1
        self.y_diff.append(self.y_values)
        while i < len(self.x_values):
            line = []
            j = 0
            while j < len(self.y_diff[i - 1]) - 1:
                result = self.y_diff[i - 1][j + 1] - self.y_diff[i - 1][j]
                line.append(result)
                j += 1
            self.y_diff.append(line)
            i += 1


class ConsoleWorker:
    x0 = 0
    y0 = 0
    type_equation = 0
    step = 0
    end_segment = 0
    accuracy = 0
    table_r = []
    okay = False

    def __int__(self):
        self.x0 = 0
        self.y0 = 0
        self.type_equation = 0
        self.step = 0
        self.end_segment = 0
        self.accuracy = 0

    def start(self):
        self.chooseType()
        self.set_start_value()
        self.set_end_segment()
        self.set_accuracy()
        self.chooseStep()

        while 1:
            try:
                self.okay = True
                eyler_solver = DiffurSolver(self.type_equation, self.x0, self.y0, self.end_segment, self.step, self.accuracy)
                eyler_solver.startEylerMethod()
                eyler_solver_2 = DiffurSolver(self.type_equation, self.x0, self.y0, self.end_segment, self.step/2, self.accuracy)
                eyler_solver_2.startEylerMethod()
                for i in range(len(eyler_solver.y_values)):
                    eyler_solver.table_value[i].append(abs((eyler_solver.y_values[i] - eyler_solver_2.y_values[2 * i])/3))
                    if abs((eyler_solver.y_values[i] - eyler_solver_2.y_values[2 * i])/3) > self.accuracy:
                        self.okay = False
                        self.step /= 2
                        break
                if self.okay:
                    print('\nMethod Modern Eyler:')
                    print_table(eyler_solver.table_value)
                    draw_graph(eyler_solver.x_values, eyler_solver.y_values, eyler_solver.type_equation, "Eyler")
                    del eyler_solver, eyler_solver_2
                    break
            except IndexError:
                print('Method Eyler not works\n')
                break

        while 1:
            try:
                self.okay = True
                adams_solver = DiffurSolver(self.type_equation, self.x0, self.y0, self.end_segment, self.step, self.accuracy)
                adams_solver.startAdamsMethod()
                adams_solver_2 = DiffurSolver(self.type_equation, self.x0, self.y0, self.end_segment, self.step/2, self.accuracy)
                adams_solver_2.startAdamsMethod()
                for i in range(len(adams_solver.y_values)):
                    adams_solver.table_value[i].append(abs((adams_solver.y_values[i] - adams_solver_2.y_values[2 * i])/15, ))
                    if abs((adams_solver.y_values[i] - adams_solver_2.y_values[2 * i])/3) > self.accuracy:
                        self.okay = False
                        self.step /= 2
                        break
                if self.okay:
                    print('\nMethod Adams:')
                    print_table(adams_solver.table_value)
                    draw_graph(adams_solver.x_values, adams_solver.y_values, adams_solver.type_equation, "Adams")
                    del adams_solver, adams_solver_2
                    break
            except IndexError:
                print('Method Adams not works\n')
                break

    def chooseType(self):
        print("Please choose a equation:\n"
              "\t1. y' = y + (1+x)y^2\n"
              "\t2. y' = e^2x + y\n"
              "\t3. y' = y/x - 3\n"
              "\t4. y' = x^2 - 2y")
        while 1:
            try:
                answer = int(input("Type: ").strip())
                if answer < 1 or answer > 4:
                    getReadyAnswer(1)
                    continue
                else:
                    self.type_equation = answer
                    break
            except ValueError:
                getReadyAnswer(1)
                continue
            except TypeError:
                getReadyAnswer(1)
                continue

    def set_start_value(self):
        print("Please input a x0 and y0\n")
        while 1:
            try:
                answer = list(input("x0 and y0: ").strip().split(" "))
                if len(answer) == 2:
                    x0 = float(answer[0].strip())
                    y0 = float(answer[1].strip())
                    self.x0 = x0
                    self.y0 = y0
                    break
                else:
                    getReadyAnswer(1)
                    continue
            except ValueError:
                getReadyAnswer(1)
                continue
            except TypeError:
                getReadyAnswer(1)
                continue

    def set_end_segment(self):
        print("Please input a end of segment\n")
        while 1:
            try:
                answer = float(input("End of segment: ").strip())
                if answer > self.x0:
                    self.end_segment = answer
                    break
                else:
                    getReadyAnswer(1)
                    continue
            except ValueError:
                getReadyAnswer(1)
                continue
            except TypeError:
                getReadyAnswer(1)
                continue

    def chooseStep(self):
        print("Please choose a step\n")
        while 1:
            try:
                answer = float(input("Step: ").strip())
                if answer > 0:
                    self.step = answer
                    break
                else:
                    getReadyAnswer(1)
                    continue
            except ValueError:
                getReadyAnswer(1)
                continue
            except TypeError:
                getReadyAnswer(1)
                continue

    def set_accuracy(self):
        print("Please input an accuracy\n")
        while 1:
            try:
                accuracy = float(input("Accuracy: ").strip())
                if accuracy > 0:
                    self.accuracy = accuracy
                    break
                else:
                    getReadyAnswer(1)
                    continue
            except ValueError:
                getReadyAnswer(1)
                continue
            except TypeError:
                getReadyAnswer(1)
                continue


def print_table(table_value):
    th = ["i", "x", "y", "y'", "R"]
    table = PrettyTable(th)
    for element in table_value:
        table.add_row(element)
    print(table)

# def print_duo_table(table_value):
#     th = ["i", "x", "y", "y'", "y''", "y'''", "y''''"]
#     table = PrettyTable(th)
#     for element in table_value:
#         table.add_row(element)
#     print(table)


def make_interpolation(solver, name):
    interpolator = Interpolator(solver.x_values, solver.y_values, 1)
    interpolator.finite_differences()
    make_graph(interpolator, solver.type_equation, name)


class DiffurSolver:
    type_equation = 0
    x = 0
    y = 0
    R = []
    end = 0
    x_values = []
    y_values = []
    step = 0
    count_of_steps = 0
    table_value = []
    accuracy = 0

    def __init__(self, types, x0, y0, endSegment, step, accuracy):
        self.type_equation = types
        self.x = x0
        self.y = y0
        self.end = endSegment
        self.step = step
        self.x_values = []
        self.y_values = []
        self.table_value = []
        self.accuracy = accuracy

    def startEylerMethod(self):
        self.count_of_steps = abs(self.end - self.x) / self.step
        i = 0
        while i <= self.count_of_steps:
            try:
                self.x_values.append(self.x)
                self.y_values.append(self.y)
                if i > 0:
                    self.R.append(round(((self.get_value_of_derivative(self.x_values[-2] + 2 * self.step, self.y) - self.y_values[-1] - self.y_values[-2])/3), 8))
                else:
                    self.R.append("-")
                self.table_value.append([i, self.x, self.y, self.get_value_of_derivative(self.x, self.y)])
                self.y = self.y + self.step/2 * (self.get_value_of_derivative(self.x, self.y) + self.get_value_of_derivative(round(self.x + self.step, 8), self.y + self.step * self.get_value_of_derivative(self.x, self.y)))
                self.x = round(self.x + self.step, 8)
                i += 1
            except ZeroDivisionError:
                self.y = self.y + self.step * self.get_value_of_derivative(self.x + 1e-9, self.y)
                self.x = round(self.x + self.step, 8)
                i += 1
                continue

    def get_value_of_derivative(self, x, y):
        if self.type_equation == 1:
            return y + (1 + x) * np.power(y, 2)
        elif self.type_equation == 2:
            return np.power(np.e, 2 * x) + y
        elif self.type_equation == 3:
            return y / x - 3
        elif self.type_equation == 4:
            return np.power(x, 2) - 2 * y

    def RungeKutta4thOrder(self, yinit, xspan, h):
        m = len(yinit)
        n = int((xspan[-1] - xspan[0]) / h)

        x = xspan[0]
        y = yinit

        xsol = np.empty((0))
        xsol = np.append(xsol, x)

        ysol = np.empty((0))
        ysol = np.append(ysol, y)

        for i in range(n):
            k1 = self.get_value_of_derivative(x, y)

            yp2 = y + k1 * (h / 2)

            k2 = self.get_value_of_derivative(x + h / 2, yp2)

            yp3 = y + k2 * (h / 2)

            k3 = self.get_value_of_derivative(x + h / 2, yp3)

            yp4 = y + k3 * h

            k4 = self.get_value_of_derivative(x + h / 2, yp4)

            for j in range(m):
                y[j] = y[j] + (h / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])

            x = x + h
            xsol = np.append(xsol, x)

            for r in range(len(y)):
                ysol = np.append(ysol, y[r])

        return [xsol, ysol]

    def ABM4thOrder(self, yinit, xspan, h):
        m = len(yinit)

        dx = int((xspan[-1] - xspan[0]) / h)

        xrk = [xspan[0] + k * h for k in range(dx + 1)]

        [xx, yy] = self.RungeKutta4thOrder(yinit, (xrk[0], xrk[3]), h)

        x = xx
        xsol = np.empty(0)
        xsol = np.append(xsol, x)

        y = yy
        yn = np.array([yy[0]])
        ysol = np.empty(0)
        ysol = np.append(ysol, y)

        for i in range(3, dx):
            x00 = x[i]
            x11 = x[i - 1]
            x22 = x[i - 2]
            x33 = x[i - 3]
            xpp = x[i] + h

            y00 = np.array([y[i]])
            y11 = np.array([y[i - 1]])
            y22 = np.array([y[i - 2]])
            y33 = np.array([y[i - 3]])

            y0prime = self.get_value_of_derivative(x00, y00)
            y1prime = self.get_value_of_derivative(x11, y11)
            y2prime = self.get_value_of_derivative(x22, y22)
            y3prime = self.get_value_of_derivative(x33, y33)

            ypredictor = y00 + (h / 24) * (55 * y0prime - 59 * y1prime + 37 * y2prime - 9 * y3prime)
            ypp = self.get_value_of_derivative(xpp, ypredictor)

            for j in range(m):
                yn[j] = y00[j] + (h / 24) * (9 * ypp[j] + 19 * y0prime[j] - 5 * y1prime[j] + y2prime[j])

            xs = x[i] + h
            xsol = np.append(xsol, xs)

            x = xsol

            for r in range(len(yn)):
                ysol = np.append(ysol, yn)

            y = ysol

        return [xsol, ysol]

    def startAdamsMethod(self):
        h = self.step
        xspan = np.array([self.x, self.end])
        yinit = np.array([self.y])

        [ts, ys] = self.ABM4thOrder(yinit, xspan, h)

        for i in range(len(ts)):
            self.table_value.append([i, round(ts[i], 8), ys[i], self.get_value_of_derivative(ts[i], ys[i])])

        self.x_values = ts
        self.y_values = ys

    def feval(self, funcName, *args):
        return eval(funcName)(*args)

    def calculate(self, x, y, step):
        try:
            return y + step * self.get_value_of_derivative(x, y)
        except ZeroDivisionError:
            return y + step * self.get_value_of_derivative(x + 1e-9, y)


print("Welcome to differential equations solver!")

while 1:
    try:
        consoleWorker = ConsoleWorker()
        consoleWorker.start()
        del consoleWorker
    except TypeError:
        getReadyAnswer(1)
        continue
    except ValueError:
        getReadyAnswer(1)
        continue
