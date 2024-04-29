from random import *

import pandas as pd
from IPython.display import display
from sympy import *


def addNoise(values, minNoise=0, maxNoise=0, superNoisesActivated=False, superNoisesMin=-8,
             superNoisesMax=8, superNoisePercent=10, noisesRelative=False, superNoisesRelative=False,
             noiseType="uniform", mean=None, stddev=None):
    # mean = 0
    # stddev = 1

    # for i in range(100):
    #   print(normalvariate(mean, stddev))
    j = 0
    nextNoise = 0
    if noiseType == "uniform":
        while j < len(values["f(x)"]):
            if superNoisesActivated and randint(0, 100) <= superNoisePercent:
                if superNoisesRelative:
                    values["f(x)"][j] += values["f(x)"][j] * uniform(superNoisesMin, superNoisesMax) / 100
                else:
                    values["f(x)"][j] += uniform(superNoisesMin, superNoisesMax)
            else:
                if noisesRelative:
                    values["f(x)"][j] += values["f(x)"][j] * uniform(minNoise, maxNoise) / 100
                else:
                    values["f(x)"][j] += uniform(minNoise, maxNoise)
            j += 1
    elif noiseType == "normal":
        while j < len(values["f(x)"]):
            if superNoisesActivated and randint(0, 100) <= superNoisePercent:
                if superNoisesRelative:
                    values["f(x)"][j] += values["f(x)"][j] * uniform(superNoisesMin, superNoisesMax) / 100
                else:
                    values["f(x)"][j] += uniform(superNoisesMin, superNoisesMax)
            else:
                if noisesRelative:
                    values["f(x)"][j] += values["f(x)"][j] * normalvariate(mean, stddev) / 100
                else:
                    values["f(x)"][j] += normalvariate(mean, stddev)
            j += 1


def generate(isRandomStep=None, maxFillEmptyChance=None, minFillEmptyChance=None, minStep=None, maxStep=None, mean=0,
             stddev=0, noiseType=None, intervalCount=None, percentToFillSegment=None, noisesRelative=False,
             superNoisesRelative=False, plot1=None, table=None, func=None, n=None, start=None, end=None, mode=None,
             minNoise=0, maxNoise=0, superNoisesActivated=False, superNoisesMin=0, superNoisesMax=0,
             superNoisePercent=0):
    symbx = symbols('x')
    # input("Enter the function of x\n")
    func = sympify(func)
    func_lambd = lambdify(symbx, func)
    # int(input("Enter the number of dots to be generated\n"))
    # float(input("Enter the initial x value\n"))
    # float(input("Enter the final x value\n"))
    step = (end - start) / n  # 1#float(input("Enter the step size\n"))
    # int(input("1 - Fixed step \n2 - Random step\n"))
    values = {"x": [], "f(x)": [], "f(x) Real": []}
    if mode == 1:
        if not isRandomStep:  # step = float(input("Enter step size\n"))
            for i in range(n):
                x = start + i * step
                try:
                    if func_lambd(x) != float("inf") and func_lambd(x) != float("-inf"):
                        values["f(x)"].append(func_lambd(x))
                        values["f(x) Real"].append(func_lambd(x))
                        values["x"].append(x)
                    else:
                        print("Infinity was excluded!")
                except ZeroDivisionError:
                    print("Infinity was excluded!")


        elif isRandomStep:
            # minStep = step/2
            # maxStep = step*2
            x = start
            for i in range(n):
                step = uniform(minStep, maxStep)
                values["x"].append(x)
                values["f(x)"].append(func_lambd(x))
                values["f(x) Real"].append(func_lambd(x))
                x += step


    elif mode == 3:
        a = intervalCount
        if not isRandomStep:
            for j in range(a):
                if uniform(0, 100) < percentToFillSegment:
                    for i in range((j - 1) * n // a, j * n // a):
                        x = start + i * step
                        values["x"].append(x)
                        values["f(x)"].append(func_lambd(x))
                        values["f(x) Real"].append(func_lambd(x))
                else:
                    fillEmptyChance = uniform(minFillEmptyChance, maxFillEmptyChance)
                    for i in range((j - 1) * n // a, j * n // a):
                        if uniform(0, 100) < fillEmptyChance:
                            x = start + i * step
                            values["x"].append(x)
                            values["f(x)"].append(func_lambd(x))
                            values["f(x) Real"].append(func_lambd(x))

        elif isRandomStep:
            x = start
            for j in range(a):
                if uniform(0, 100) < percentToFillSegment:
                    for i in range((j - 1) * n // a, j * n // a):
                        step = uniform(minStep, maxStep)
                        x += step
                        values["x"].append(x)
                        values["f(x)"].append(func_lambd(x))
                        values["f(x) Real"].append(func_lambd(x))
                else:
                    fillEmptyChance = uniform(minFillEmptyChance, maxFillEmptyChance)
                    for i in range((j - 1) * n // a, j * n // a):
                        step = uniform(minStep, maxStep)
                        x += step
                        if uniform(0, 100) < fillEmptyChance:
                            values["x"].append(x)
                            values["f(x)"].append(func_lambd(x))
                            values["f(x) Real"].append(func_lambd(x))

    addNoise(mean=mean, stddev=stddev, noiseType=noiseType, noisesRelative=noisesRelative,
             superNoisesRelative=superNoisesRelative, values=values, minNoise=minNoise, maxNoise=maxNoise,
             superNoisesActivated=superNoisesActivated, superNoisesMin=superNoisesMin, superNoisesMax=superNoisesMax,
             superNoisePercent=superNoisePercent)
    table.delete(*table.get_children())
    for i in range(len(values["x"])):
        table.insert("", "end", values=["{:.4f}".format(values["x"][i]), "{:.4f}".format(values["f(x) Real"][i]),
                                        "{:.4f}".format(values["f(x)"][i])])
    if plot1 is not None:
        plot1.cla()
        plot1.plot(values["x"], values["f(x)"], 'ro', label='Dots with noises', marker='.')
    df = pd.DataFrame(data=values)
    display(df)
    writer = pd.ExcelWriter('../dots.xlsx')
    df.to_excel(writer)
    worksheet = writer.sheets['Sheet1']
    worksheet.cell(row=1, column=1).value = "f(x)=" + " " + str(func)
    writer._save()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
