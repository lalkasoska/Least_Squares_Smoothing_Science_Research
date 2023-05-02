import copy
import math
import os
import random
import numpy as np
import matplotlib as mpl  # Библиотека для построения графиков
import matplotlib.pyplot as plt
from sympy import *

import GeneticDot
import GeneticVector
from OtherMethods import *

mpl.rcParams['figure.figsize'] = (12, 8)  # Задаём параметры окна с графиками
mpl.rcParams['axes.grid'] = True

working = True


def equations(consts):
    # a0, a1, a2 = variables
    shoulder_length, parabola_start, parabola_end, x, y, yx, yx2, x2, x3, x4 = consts
    parabola_start = int(parabola_start)
    parabola_end = int(parabola_end) + 1

    x_sum = sum(x[parabola_start:parabola_end])
    x2_sum = sum(x2[parabola_start:parabola_end])
    x3_sum = sum(x3[parabola_start:parabola_end])
    x4_sum = sum(x4[parabola_start:parabola_end])
    y_sum = sum(y[parabola_start:parabola_end])
    yx_sum = sum(yx[parabola_start:parabola_end])
    yx2_sum = sum(yx2[parabola_start:parabola_end])

    # return shoulder_length * 2 * a0 + a1 * x_sum + a2 * x2_sum - y_sum, \
    #        a0 * x_sum + a1 * x2_sum + a2 * x3_sum - yx_sum, \
    #        a0 * x2_sum + a1 * x3_sum + a2 * x4_sum - yx2_sum
    return np.array(
        [[shoulder_length * 2 + 1, x_sum, x2_sum], [x_sum, x2_sum, x3_sum], [x2_sum, x3_sum, x4_sum]]), np.array(
        [y_sum, yx_sum, yx2_sum])


def smooth(x, y, y_real, yx, yx2, x2, x3, x4, func_lambd, table=None, path='dots.xlsx', shoulder_lengthh=10,
           smoothEdges=False, showRealFunction=False, countEdges=True,
           deviationType="Squares", realtime=False, savepath=None, plot1=None, plot2=None, compareWithNoised=False,
           func="NoFunction", comparing=False, sideShoulderType="Increment", linePlot=None):
    #z = 0
    title = ""
    table.delete(*table.get_children())
    needToRevert = False
    shoulder_length = shoulder_lengthh
    y_smoothed = copy.deepcopy(y)
    shoulders = []
    n = len(x)
    for i in range(0, n):
        if smoothEdges:
            if sideShoulderType == "Increment":
                if not (i == 0 or i == n - 1):
                    if i < shoulder_length and i > 0:
                        if smoothEdges:
                            shoulder_length = i
                            needToRevert = True
                    if n - i - 1 < shoulder_length and n - i - 1 > 0:
                        if smoothEdges:
                            shoulder_length = n - i - 1  # !!!!!!!!!!!!!!!!!
                            needToRevert = True
                    parabola_start = i - shoulder_length
                    parabola_end = i + shoulder_length


            elif sideShoulderType == "Amortization":
                if i < shoulder_length and i >= 0:
                    if smoothEdges:
                        parabola_start = 0
                        parabola_end = shoulder_length * 2
                elif n - i - 1 < shoulder_length and n - i - 1 >= 0:
                    if smoothEdges:
                        parabola_start = n - (shoulder_length * 2) - 1
                        parabola_end = n - 1
                else:
                    parabola_start = i - shoulder_length
                    parabola_end = i + shoulder_length
                print(f"diff:{parabola_end - parabola_start}")
        elif not (i < shoulder_length and i >= 0) and not (n - i - 1 < shoulder_length and n - i - 1 >= 0):
            parabola_start = i - shoulder_length
            parabola_end = i + shoulder_length
        else:
            table.insert("", "end", values=["{:.4f}".format(x[i]), "{:.4f}".format(y_real[i]), "{:.4f}".format(y[i]),
                                            "{:.4f}".format(y_smoothed[i]),
                                            "{:.4f}".format(abs(y_real[i] - y[i])),
                                            "{:.4f}".format(abs(y_real[i] - y_smoothed[i])), "{:.4f}".format(
                    abs(abs(y_real[i] - y[i]) - abs(y_real[i] - y_smoothed[i]))), "-"])
            shoulders.append(0)
            continue

        if not ((i == 0 or i == n - 1) and sideShoulderType == "Increment"):
            data = (shoulder_length, parabola_start, parabola_end, x, y, yx, yx2, x2, x3, x4)
            try:
                a0, a1, a2 = np.linalg.solve(*equations(data))
            except:
                a0, a1, a2 = np.linalg.lstsq(*equations(data))[0]
            # print(f"start:{parabola_start}, end:{parabola_end}")
            #print(f"parabola_start:{parabola_start}, parabola_end:{parabola_end}, i:{i}")
            xspace10 = np.linspace(x[parabola_start], x[parabola_end], 500)
            xspace = np.linspace(x[0], x[-1], 500)
            fragment_smoothed = [a0 + a1 * xi + a2 * xi ** 2 for xi in xspace10]
            y_smoothed[i] = 0
            y_smoothed[i] = a0 + a1 * x[i] + a2 * x[i] ** 2
            table.insert("", "end", values=["{:.4f}".format(x[i]), "{:.4f}".format(y_real[i]), "{:.4f}".format(y[i]),
                                            "{:.4f}".format(y_smoothed[i]), "{:.4f}".format(abs(y_real[i] - y[i])),
                                            "{:.4f}".format(abs(y_real[i] - y_smoothed[i])), "{:.4f}".format(
                    abs(abs(y_real[i] - y[i]) - abs(y_real[i] - y_smoothed[i]))), shoulder_length])
            if realtime:
                plt.clf()
                plt.ylim(min(y) - abs(min(y))*0.4, max(y) + abs(max(y))*0.4)
                plt.plot(x, y, 'ro',alpha=0.2, label='Точки с шумом')
                # plt.plot(xspace,y_real, 'g', label='real_function')
                if linePlot:
                    plt.plot(x[:i+1], y_smoothed[:i+1], 'b', label='Сглаженные точки', linewidth=2)
                else:
                    plt.plot(x[:i+1], y_smoothed[:i+1], 'bo', marker='.', label='Сглаженные точки')

                if i != len(x)-1:
                    plt.plot(x[parabola_start:parabola_end + 1], y[parabola_start:parabola_end + 1],
                             'ro',alpha=0.4, label="Точки опоры")
                    plt.plot(xspace10, fragment_smoothed, 'darkblue', label='Сглаживающая парабола',linewidth=3)
                    plt.plot(x[i], y_smoothed[i], 'bo', label="Текущая точка",markersize=13)
                plt.legend(loc='upper left')
                plt.title("Плечо: " + str(shoulder_length) + ", Текущая точка: " + str(i))
                plt.pause(0.0000001)
                plt.show(block=False)
                # if z==0:
                #     import time
                #     time.sleep(10)
                #     z+=1
        if (i == 0 or i == n - 1) and sideShoulderType == "Increment":
            table.insert("", "end", values=["{:.4f}".format(x[i]), "{:.4f}".format(y_real[i]), "{:.4f}".format(y[i]),
                                            "{:.4f}".format(y_smoothed[i]),
                                            "{:.4f}".format(abs(y_real[i] - y[i])),
                                            "{:.4f}".format(abs(y_real[i] - y_smoothed[i])), "{:.4f}".format(
                    abs(abs(y_real[i] - y[i]) - abs(y_real[i] - y_smoothed[i]))), "-"])
            shoulders.append(0)
        else:
            shoulders.append(shoulder_length)
        if needToRevert:
            shoulder_length = shoulder_lengthh

    # plt.show()

    deviation = 0
    deviationyKernel = 0
    deviationyGauss = 0
    deviationyTrees = 0
    deviationyForest = 0
    deviationyMLP = 0
    deviationyNeighbours = 0

    if comparing:
        x = np.array(x).reshape(-1, 1)
        y = np.array(y)
        yKernel = kernel(x, y)
        yGauss = gauss(x, y)
        yTrees = regTrees(x, y)
        yForest = randomForest(x, y)
        yMLP = MLP(x, y)
        yNeighbours = kNeighbours(x, y)

    # else:
    #     yKernel = y
    #     yGauss = y
    #     yTrees = y
    #     yForest = y
    #     yMLP=y
    #     yNeighbours = y
    if comparing:
        for i in range(len(y)):
            deviationyKernel += (y_real[i] - yKernel[i]) ** 2
        for i in range(len(y)):
            deviationyGauss += (y_real[i] - yGauss[i]) ** 2
        for i in range(len(y)):
            deviationyTrees += (y_real[i] - yTrees[i]) ** 2
        for i in range(len(y)):
            deviationyForest += (y_real[i] - yForest[i]) ** 2
        for i in range(len(y)):
            deviationyMLP += (y_real[i] - yMLP[i]) ** 2
        for i in range(len(y)):
            deviationyNeighbours += (y_real[i] - yNeighbours[i]) ** 2
    if countEdges:
        if deviationType == "Squares":
            for i in range(len(y)):
                deviation += (y_real[i] - y_smoothed[i]) ** 2


        # ("shoulder_lengthh:", shoulder_lengthh)
        # ("Error:", deviation)
        elif deviationType == "AverageSquares":
            for i in range(len(y)):
                deviation += (y_real[i] - y_smoothed[i]) ** 2
            deviation /= len(y)
            deviation = math.sqrt(deviation)
        # ("shoulder_lengthh:", shoulder_lengthh)
        # ("Error:", deviation)
        elif deviationType == "Absolute":
            for i in range(len(y)):
                deviation += abs(y_real[i] - y_smoothed[i])
        # ("shoulder_lengthh:", shoulder_lengthh)
        # ("Error:", deviation)
    else:
        if deviationType == "Squares":
            for i in range(shoulder_lengthh, len(y) - shoulder_lengthh):
                deviation += (y_real[i] - y_smoothed[i]) ** 2
        # ("shoulder_lengthh:", shoulder_lengthh)
        # ("Error:", deviation)

        elif deviationType == "AverageSquares":
            for i in range(shoulder_lengthh, len(y) - shoulder_lengthh):
                deviation += (y_real[i] - y_smoothed[i]) ** 2
            deviation /= (len(y) - 2 * shoulder_lengthh)
            deviation = math.sqrt(deviation)
        # ("shoulder_lengthh:", shoulder_lengthh)
        # ("Error:", deviation)
        elif deviationType == "Absolute":
            for i in range(shoulder_lengthh, len(y) - shoulder_lengthh):
                deviation += abs(y_real[i] - y_smoothed[i])
        # ("shoulder_lengthh:", shoulder_lengthh)
        # ("Error:", deviation)
    if comparing:
        print(f"MyFunc deviation: {deviation}")
        print(f"Kernel deviation: {deviationyKernel}")
        print(f"Gauss deviation: {deviationyGauss}")
        print(f"Trees deviation: {deviationyTrees}")
        print(f"Forest deviation: {deviationyForest}")
        print(f"MLP deviation: {deviationyMLP}")
        print(f"kNeighbours deviation: {deviationyNeighbours}")
        from tkinter import filedialog
        with open(filedialog.asksaveasfilename(initialdir="SuperComparison\Fix") + ".txt", 'w+') as f:
            f.write(f"MyFunc deviation: {deviation}\n"
                    f"Kernel deviation: {deviationyKernel}\n"
                    f"Gauss deviation: {deviationyGauss}\n"
                    f"Trees deviation: {deviationyTrees}\n"
                    f"Forest deviation: {deviationyForest}\n"
                    f"MLP deviation: {deviationyMLP}\n"
                    f"kNeighbours deviation: {deviationyNeighbours}\n\n")
    if plot1 is not None:
        plot1.cla()
        shoulderVectorStandardized = [a * ((max(y) - min(y)) / 2 / max(shoulders)) - max(y) * 2 for a in
                                      shoulders]
        # plot1.plot(x, shoulderVectorStandardized, "gray")
        # plot1.scatter(x[np.argmax(shoulderVectorStandardized)], max(shoulderVectorStandardized), color="black")
        # plot1.text(x[np.argmax(shoulderVectorStandardized)], max(shoulderVectorStandardized), f"Max: {max(shoulders)}")
        # plot2.bar(x, shoulders, width=1)
    if plot1 is not None:
        # plot1.cla()
        plot1.plot(x, y, 'ro', alpha=0.2, label='Точки с шумом')
        if showRealFunction:
            y_realDrawn = [func_lambd(xi) for xi in xspace]
            plot1.plot(xspace, y_realDrawn, 'g', label='Real function')
        # plt.plot(xspace10,fragment_smoothed, 'b', label='Parabola')
        # plt.plot(xspace, fragment_smoothed, 'b', label='Parabola')
        if linePlot:
            plot1.plot(x, y_smoothed, 'b', label='Сглаженные точки', linewidth=2)
        else:
            plot1.plot(x, y_smoothed, 'bo', marker='.', label='Сглаженные точки')
        if comparing:
            plot1.plot(x, yKernel, color='k', label='Kernel')
            # plt.plot(x, local(x,y), color='blue', linestyle='--',linewidth=3.0, label='Local')

            plot1.plot(x, yGauss, color='r', label='Gauss')

            # plot1.plot(x, yTrees, color='orange',  label='Regression Tree')

            print(yKernel)
            print(yForest)
            plot1.plot(x, yForest, color='y', label='Random Forest')
            plot1.plot(x, yMLP, color='orange', label='Neural Network')
            plot1.plot(x, yNeighbours, color='purple', label='k Nearest Neighbours')
        plot1.legend(loc='upper left')
        title = "\nплечо: " + str(shoulder_lengthh) + ", отклонение: " + "{:.4f}".format(deviation)
    # if savepath is not None and savepath != "" and comparing:
    #     # fig2 = mpl.figure.Figure(figsize=(8, 5), dpi=100)
    #     # fig2plot = fig2.add_subplot(111)
    #     # fig2plot.plot(x, y, 'ro', label='Точки с шумом', marker='.')
    #     # if showRealFunction:
    #     #     y_real = [func_lambd(xi) for xi in xspace]
    #     #     plot1.plot(xspace, y_real, 'g', label='Реальная функция')
    #     # # plt.plot(xspace10,fragment_smoothed, 'b', label='Parabola')
    #     # fig2plot.plot(x, y_smoothed, 'bo', marker='.', label='Сглаженные точки')
    #     # fig2plot.legend(loc='upper left')
    #     # fig2plot.text(x[n // 2], y[n // 2] * 2, "Плечо: " + str(shoulder_lengthh))
    #     savepath = savepath.replace("\n", "")
    #     if not os.path.exists(savepath):
    #         os.mkdir(savepath)
    #     plot1.savefig(savepath + "/" + str(shoulder_lengthh) + ".png")
    deviationFromNoised = None
    if compareWithNoised:
        deviationFromNoised = sum([abs(y0 - y1) for y0, y1 in zip(y, y_smoothed)])

    return plt, deviation, title, deviationFromNoised


def findBestShoulder(x, y, y_real, yx, yx2, x2, x3, x4, func_lambd, smoothEdges=True, deviationType="Squares",
                     countEdges=True, realtime=False,
                     progress=None, window=None, savepath=None, table=None, alg=None, dichotomyMin=None,
                     dichotomyMax=None, compareWithNoised=False, shoulderLimit=None, showGraphs=True, plot21=None,
                     plot22=None, sideShoulderType=None, goldenMin=None, goldenMax=None, tau=None):
    shoulderLimit = min(shoulderLimit, len(x) // 2 - 1)
    global working
    bestShoulder = -1
    minDeviation = float("inf")
    count = 0
    deviations = []
    deviationsFromNoised = []
    shoulders = []
    fig, axs = plt.subplots(2)
    if shoulderLimit is None:
        shoulderLimit = len(x) // 2 - 1
    if alg == "Loop":
        for shoulder in range(1, shoulderLimit + 1):
            if working:
                if progress is not None:
                    progress.config(value=100 * shoulder / shoulderLimit)
                    if window is not None:
                        window.update()
                shoulders.append(shoulder)
                _, deviation, title, deviationFromNoised = smooth(x, y, y_real, yx, yx2, x2, x3, x4, func_lambd,
                                                                  table=table, shoulder_lengthh=shoulder,
                                                                  smoothEdges=smoothEdges,
                                                                  deviationType=deviationType, countEdges=countEdges,
                                                                  savepath=savepath,
                                                                  compareWithNoised=compareWithNoised,
                                                                  sideShoulderType=sideShoulderType)
                deviations.append(deviation)
                deviationsFromNoised.append(deviationFromNoised)
                count += 1
                if deviation < minDeviation:
                    count = 0
                    minDeviation = deviation
                    bestShoulder = shoulder
                # if count == 30:
                # break
                if realtime:
                    axs[0].cla()
                    axs[0].set_title("Отклонение от реальных точек")
                    axs[0].plot(shoulders, deviations, 'g', label="Зависимость ошибки от плеча")
                    axs[0].scatter(bestShoulder, minDeviation, s=30)
                    axs[0].text(bestShoulder, minDeviation, "Оптимальное плечо:" + str(bestShoulder))

                    axs[1].cla()
                    axs[1].set_title("Отклонение от зашумлённых точек")
                    axs[1].plot(shoulders, deviationsFromNoised, 'r', label="Отклонение от зашумлённых точек")
                    axs[1].scatter(bestShoulder, deviationsFromNoised[bestShoulder - 1], s=30)
                    axs[1].text(bestShoulder, deviationsFromNoised[bestShoulder - 1],
                                "Оптимальное плечо:" + str(bestShoulder))

                    #plt.pause(0.0001)
                    fig.show()
    elif alg == "Dichotomy":
        log2length = float(log(dichotomyMax - dichotomyMin, 2))
        print(log2length)
        eps = 3.1
        delta = 1
        a = dichotomyMin
        b = dichotomyMax
        while abs(b - a) >= eps:
            progress.config(value=(log2length - float(log(abs(b - a), 2)) + 1) * 100 / log2length)
            window.update()

            print(f"a: {a}, b: {b}")
            lamb = (a + b) // 2 - delta
            mu = (a + b) // 2 + delta
            print(f"lamb: {lamb}, mu: {mu}")

            _, flamb, _, _ = smooth(x, y, y_real, yx, yx2, x2, x3, x4, func_lambd, table=table,
                                    shoulder_lengthh=lamb, smoothEdges=smoothEdges,
                                    deviationType=deviationType, countEdges=countEdges, savepath=savepath,
                                    sideShoulderType=sideShoulderType)
            _, fmu, _, _ = smooth(x, y, y_real, yx, yx2, x2, x3, x4, func_lambd, table=table,
                                  shoulder_lengthh=mu, smoothEdges=smoothEdges,
                                  deviationType=deviationType, countEdges=countEdges, savepath=savepath,
                                  sideShoulderType=sideShoulderType)
            if flamb < fmu:
                b = mu
            else:
                a = lamb
        bestShoulder = (a + b) // 2 + 1
    elif alg == "Golden":
        log2length = float(log(goldenMax - goldenMin, 2))
        # a = 0
        # b = 8
        eps = 3.1
        ai = goldenMin
        bi = goldenMax
        tau = 0.618
        lambdi = round(ai + (1 - tau) * (bi - ai))
        mui = round(ai + tau * (bi - ai))
        while (bi - ai) >= eps:
            if progress is not None:
                progress.config(value=(log2length - float(log(abs(bi - ai), 2)) + 1) * 100 / log2length)
            window.update()
            _, flambdi, _, _ = smooth(x, y, y_real, yx, yx2, x2, x3, x4, func_lambd, table=table,
                                      shoulder_lengthh=lambdi, smoothEdges=smoothEdges,
                                      deviationType=deviationType, countEdges=countEdges, savepath=savepath,
                                      sideShoulderType=sideShoulderType)
            _, fmui, _, _ = smooth(x, y, y_real, yx, yx2, x2, x3, x4, func_lambd, table=table,
                                   shoulder_lengthh=mui, smoothEdges=smoothEdges,
                                   deviationType=deviationType, countEdges=countEdges, savepath=savepath,
                                   sideShoulderType=sideShoulderType)
            if flambdi > fmui:
                ai = lambdi
                bi = bi
                lambdi = mui
                mui = round(ai + tau * (bi - ai))
            else:
                ai = ai
                bi = mui
                mui = lambdi
                lambdi = round(ai + (1 - tau) * (bi - ai))
        bestShoulder = (ai + bi) // 2 + 1
    # fig.suptitle('Vertically stacked subplots')
    # axs[0].plot(x, y)
    # axs[1].plot(x, -y)

    # plt.cla()
    if alg == "Loop":
        axs[0].set_title("Отклонение от реальных точек")
        axs[0].plot(shoulders, deviations, 'g', label="Зависимость ошибки от плеча")
        axs[0].scatter(bestShoulder, minDeviation, s=30)
        axs[0].text(bestShoulder, minDeviation, "Оптимальное плечо:" + str(bestShoulder))

        plot21.set_title("Отклонение от реальных точек")
        plot21.plot(shoulders, deviations, 'g', label="Зависимость ошибки от плеча")
        plot21.scatter(bestShoulder, minDeviation, s=30)
        plot21.text(bestShoulder, minDeviation, str(bestShoulder))
        if compareWithNoised:
            # plt.cla()
            axs[1].set_title("Отклонение от зашумлённых точек")
            axs[1].plot(shoulders, deviationsFromNoised, 'r', label="Отклонение от зашумлённых точек")
            axs[1].scatter(bestShoulder, deviationsFromNoised[bestShoulder - 1], s=30)
            axs[1].text(bestShoulder, deviationsFromNoised[bestShoulder - 1], "Оптимальное плечо:" + str(bestShoulder))

            plot22.set_title("Отклонение от зашумлённых точек")
            plot22.plot(shoulders, deviationsFromNoised, 'r', label="Отклонение от зашумлённых точек")
            plot22.scatter(bestShoulder, deviationsFromNoised[bestShoulder - 1], s=30)
            plot22.text(bestShoulder, deviationsFromNoised[bestShoulder - 1], str(bestShoulder))
            # axs[1].show(block=False)
            # axs[1].savefig(savepath + "/DistanceFromNoised.png")
        if showGraphs: fig.show()

    if savepath is not None and alg == "Loop":
        savepath = savepath.replace("\n", "")
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        fig.savefig(savepath + "/ShouldersErrors.png")
    if progress is not None:
        progress.config(value=100)
    return bestShoulder


def findBestShoulderForDot(x, y, y_real, yx, yx2, x2, x3, x4, func_lambd, table, smoothEdges, countEdges, realtime,
                           savepath=None, iterations=None, shoulderLimit=math.inf, shoulderChangeLimit=math.inf,
                           mode="Loop", accuracy=None, dots=None, index=0, deviationType="Squares", populationSize=None,
                           mutationProb=None):
    if index != 0 and index < len(x) - 1:
        minDeviation = float("inf")
        #  for i in range(iterations):
        # if i > min(len(x) - index, index):
        #   break
        shoulders = []
        shoulderLimit = min(len(x) - index - 1, index, shoulderLimit)
        bestShoulder = 1
        if mode == "Loop":
            for shoulder_length in range(1, shoulderLimit+1):
                # shoulder_length = random.randint(1, min(len(x) - index, index))
                # while shoulder_length in shoulders:
                #    shoulder_length = random.randint(1,min(len(x)-index,index))

                parabola_start = index - shoulder_length
                parabola_end = index + shoulder_length
                data = (shoulder_length, parabola_start, parabola_end, x, y, yx, yx2, x2, x3, x4)
                try:
                    a0, a1, a2 = np.linalg.solve(*equations(data))
                except:
                    a0, a1, a2 = np.linalg.lstsq(*equations(data))[0]

                dot_smoothed = a0 + a1 * x[index] + a2 * x[index] ** 2

                if deviationType == "Squares":
                    deviation = (dot_smoothed - y_real[index]) ** 2
                    print(index,shoulder_length, deviation, dot_smoothed, y_real[index], sep='|')

                # ("shoulder_lengthh:", shoulder_lengthh)
                # ("Error:", deviation)
                elif deviationType == "AverageSquares":
                    deviation = (dot_smoothed - y_real[index]) ** 2
                    deviation /= len(y)
                    deviation = math.sqrt(deviation)
                # ("shoulder_lengthh:", shoulder_lengthh)
                # ("Error:", deviation)
                elif deviationType == "Absolute":
                    deviation = abs(dot_smoothed - y_real[index])
                # ("shoulder_lengthh:", shoulder_lengthh)
                # ("Error:", deviation)
                # ("x:", x[index])
                # ("shld: ", shoulder_length)
                # ("dev: ", deviation)
                if deviation < minDeviation:
                    minDeviation = deviation
                    bestShoulder = shoulder_length
        elif mode == "Random":
            for i in range(iterations):
                shoulder_length = random.randint(1, min(min(len(x) - index - 1, index), shoulderLimit))
                while shoulder_length in shoulders:
                    shoulder_length = random.randint(1, min(min(len(x) - index - 1, index), shoulderLimit))
                parabola_start = index - shoulder_length
                parabola_end = index + shoulder_length
                data = (shoulder_length, parabola_start, parabola_end, x, y, yx, yx2, x2, x3, x4)
                try:
                    a0, a1, a2 = np.linalg.solve(*equations(data))
                except:
                    a0, a1, a2 = np.linalg.lstsq(*equations(data))[0]
                dot_smoothed = a0 + a1 * x[index] + a2 * x[index] ** 2
                # print("y=",y[index],"yreal=", y_real[index])
                if deviationType == "Squares":
                    deviation = (dot_smoothed - y_real[index]) ** 2
                # ("shoulder_lengthh:", shoulder_lengthh)
                # ("Error:", deviation)
                elif deviationType == "AverageSquares":
                    deviation = (dot_smoothed - y_real[index]) ** 2
                    deviation /= len(y)
                    deviation = math.sqrt(deviation)
                # ("shoulder_lengthh:", shoulder_lengthh)
                # ("Error:", deviation)
                elif deviationType == "Absolute":
                    deviation = abs(dot_smoothed - y_real[index])
                # ("x:", x[index])
                # ("shld: ", shoulder_length)
                # ("dev: ", deviation)
                if deviation < minDeviation:
                    minDeviation = deviation
                    bestShoulder = shoulder_length
                if deviation <= accuracy:
                    break
        elif mode == "Genetic":
            bestShoulder = GeneticDot.genetic(index, populationSize, iterations, mutationProb, x=x, y=y, y_real=y_real,
                                              yx=yx, yx2=yx2, x2=x2, x3=x3, x4=x4, func_lambd=func_lambd, table=table,
                                              smoothEdges=smoothEdges, deviationType=deviationType,
                                              countEdges=countEdges, realtime=realtime, savepath=None,
                                              shoulderLimit=shoulderLimit)
        elif mode == "Dichotomy":
            dichotomyMin = 0
            dichotomyMax = min(min(len(x) - index - 1, index), shoulderLimit)
            log2length = float(log(dichotomyMax - dichotomyMin, 2))
            print(log2length)
            eps = 3.1
            delta = 1
            a = dichotomyMin
            b = dichotomyMax
            while abs(b - a) >= eps:
                # progress.config(value=(log2length - float(log(abs(b - a), 2)) + 1) * 100 / log2length)
                # window.update()

                print(f"a: {a}, b: {b}")
                lamb = (a + b) // 2 - delta
                mu = (a + b) // 2 + delta
                print(f"lamb: {lamb}, mu: {mu}")

                parabola_start = index - lamb
                parabola_end = index + lamb
                data = (lamb, parabola_start, parabola_end, x, y, yx, yx2, x2, x3, x4)
                try:
                    a0, a1, a2 = np.linalg.solve(*equations(data))
                except:
                    a0, a1, a2 = np.linalg.lstsq(*equations(data))[0]
                dot_smoothed = a0 + a1 * x[index] + a2 * x[index] ** 2
                # print("y=",y[index],"yreal=", y_real[index])
                flamb = abs(dot_smoothed - y_real[index])

                parabola_start = index - mu
                parabola_end = index + mu
                data = (mu, parabola_start, parabola_end, x, y, yx, yx2, x2, x3, x4)
                try:
                    a0, a1, a2 = np.linalg.solve(*equations(data))
                except:
                    a0, a1, a2 = np.linalg.lstsq(*equations(data))[0]
                dot_smoothed = a0 + a1 * x[index] + a2 * x[index] ** 2
                # print("y=",y[index],"yreal=", y_real[index])
                fmu = abs(dot_smoothed - y_real[index])

                if flamb < fmu:
                    b = mu
                else:
                    a = lamb
            bestShoulder = (a + b) // 2
        return bestShoulder


def findBestShoulderVector(x, y, y_real, yx, yx2, x2, x3, x4, func_lambd, table=None, path='dots.xlsx', showGraphs=True,
                           smoothEdges=True, deviationType="Squares", countEdges=True, realtime=False,
                           progress=None, window=None, savepath=None, iterations=20, mode="Loop", accuracy=None,
                           generateWhole=True, populationSize=None, mutationProb=None, shoulderLimit=None,
                           compareWithNoised=False, plot21=None, plot22=None):
    shoulderLimit = min(shoulderLimit, len(x) // 2)
    global working
    bestShoulderVector = []
    minDeviation = float("inf")
    deviations = []
    shoulders = []
    fig, axs = plt.subplots(2)

    if generateWhole:
        if mode == "Genetic":
            # print(populationSize)
            bestShoulderVector, deviations, deviationsFromNoised = GeneticVector.genetic(progress, window,
                                                                                         populationSize, iterations,
                                                                                         mutationProb, len(x),
                                                                                         shoulderLimit=shoulderLimit,
                                                                                         table=table,
                                                                                         smoothEdges=smoothEdges,
                                                                                         deviationType=deviationType,
                                                                                         countEdges=countEdges,
                                                                                         realtime=realtime,
                                                                                         savepath=None, x=x, y=y,
                                                                                         y_real=y_real, yx=yx, yx2=yx2,
                                                                                         x2=x2, x3=x3, x4=x4,
                                                                                         func_lambd=func_lambd)
        elif mode == "Random":
            for i in range(iterations):
                if working:
                    if window is not None:
                        window.update()
                    randomVector = []
                    for j in range(len(x)):
                        if j == 0:
                            randomVector.append(0)
                        elif j == 1:
                            randomVector.append(1)
                        elif len(x) - j - 1 == 0:
                            randomVector.append(0)
                        else:
                            randomVector.append(random.randint(1, min(j, len(x) - j - 1)))

                    deviation, title, _ = smoothSeparate(x, y, y_real, yx, yx2, x2, x3, x4, func_lambd, table=table,
                                                            shoulderVector=randomVector, smoothEdges=smoothEdges,
                                                            deviationType=deviationType, countEdges=countEdges,
                                                            realtime=realtime, savepath=None)
                    if deviation < minDeviation:
                        bestShoulderVector = randomVector
                        minDeviation = deviation
    else:

        for i in range(len(x)):
            if working:
                if window is not None:
                    window.update()
                progress.config(value=i * 100 / len(x))
                bestShoulderVector.append(
                    findBestShoulderForDot(x, y, y_real, yx, yx2, x2, x3, x4, func_lambd, table=table,
                                           smoothEdges=smoothEdges, deviationType=deviationType, countEdges=countEdges,
                                           realtime=realtime, dots=None, index=i,
                                           iterations=iterations, mode=mode, accuracy=accuracy,
                                           populationSize=populationSize, shoulderLimit=shoulderLimit,
                                           mutationProb=mutationProb))
    progress.config(value=100)

    # axs[0].set_title("Отклонение от реальных точек")
    # axs[0].plot(range(len(deviations)), deviations, 'g', label="Зависимость ошибки от плеча")
    # axs[0].scatter(bestShoulder, minDeviation, s=30)
    # axs[0].text(bestShoulder, minDeviation, "Оптимальное плечо:" + str(bestShoulder))
    #
    # if compareWithNoised:
    #     plt.cla()
    #     print(deviationsFromNoised)
    #     axs[1].set_title("Отклонение от зашумлённых точек")
    #     axs[1].plot(range(len(deviationsFromNoised)), deviationsFromNoised, 'r', label="Отклонение от зашумлённых точек")
    #     axs[1].scatter(bestShoulder, deviationsFromNoised[bestShoulder - 1], s=30)
    #     axs[1].text(bestShoulder, deviationsFromNoised[bestShoulder - 1], "Оптимальное плечо:" + str(bestShoulder))
    #     axs[1].show(block=False)
    #     axs[1].savefig(savepath + "/DistanceFromNoised.png")
    # if showGraphs: fig.show()
    return bestShoulderVector


def smoothSeparate(x, y, y_real, yx, yx2, x2, x3, x4, func_lambd, shoulderVector, smoothEdges=False,
                   showRealFunction=False, countEdges=False,
                   deviationType="Squares", realtime=False, savepath=None, table=None, plot1=None, plot2=None,
                   func="NoFunc", compareWithNoised=False, amortizeSides=None, sideLength=None, comparing=None,linePlot=None):
    # print("щьз",compareWithNoised)

    table.delete(*table.get_children())
    shoulderVector[0] = 0
    shoulderVector[-1] = 0
    symbx = symbols('x')
    y_smoothed = copy.deepcopy(y)
    n = len(shoulderVector)
    for i in range(0, len(shoulderVector)):

        if shoulderVector[i] < 1: shoulderVector[i] = 1
        # while shoulderVector[i] > min(len(shoulderVector)-1-i,i): shoulderVector[i]-=1
        shoulder_length = shoulderVector[i]
        if amortizeSides:
            if i < sideLength:
                if smoothEdges:
                    parabola_start = 0
                    parabola_end = sideLength * 2
            elif n - i - 1 < sideLength:
                if smoothEdges:
                    parabola_start = n - (sideLength * 2) - 1
                    parabola_end = n - 1
            else:
                parabola_start = i - shoulder_length
                parabola_end = i + shoulder_length
        else:
            parabola_start = i - shoulder_length
            parabola_end = i + shoulder_length
        data = (shoulder_length, parabola_start, parabola_end, x, y, yx, yx2, x2, x3, x4)
        try:
            a0, a1, a2 = np.linalg.solve(*equations(data))
        except:
            a0, a1, a2 = np.linalg.lstsq(*equations(data))[0]

        y_smoothed[i] = a0 + a1 * x[i] + a2 * x[i] ** 2
        table.insert("", "end", values=["{:.4f}".format(x[i]), "{:.4f}".format(y_real[i]), "{:.4f}".format(y[i]),
                                        "{:.4f}".format(y_smoothed[i]), "{:.4f}".format(abs(y_real[i] - y[i])),
                                        "{:.4f}".format(abs(y_real[i] - y_smoothed[i])),
                                        "{:.4f}".format(abs(abs(y_real[i] - y[i]) - abs(y_real[i] - y_smoothed[i]))),
                                        shoulder_length])

        # if realtime:
        #     xspace10 = np.linspace(x[parabola_start], x[parabola_end], 500)
        #     fragment_smoothed = [a0 + a1 * xi + a2 * xi ** 2 for xi in xspace10]
        #     plt.clf()
        #
        #     plt.plot(x, y, 'ro', label='Точки с шумом', marker='.')
        #     # plt.plot(xspace,y_real, 'g', label='real_function')
        #     plt.plot(xspace10, fragment_smoothed, 'black', label='Сглаживающая парабола')
        #     plt.plot(x[:i + 1], y_smoothed[:i + 1], 'bo', marker='.', label="Сглаженные точки")
        #
        #     plt.plot(x[i - shoulder_length:i + shoulder_length + 1], y[i - shoulder_length:i + shoulder_length + 1],
        #              'ro', label="Точки опоры")
        #     plt.plot(x[i], y_smoothed[i], 'bo', label="Текущая точка")
        #     plt.legend(loc='upper left')
        #
        #     # plt.plot(x, y, 'ro', label='Dots with noises', marker='.')
        #     # plt.plot(xspace10, fragment_smoothed, 'b', label='Parabola')
        #     # plt.plot(x, y_smoothed, 'bo', marker='.', label="Smoothed dots")
        #     # plt.legend(loc='upper left')
        #     # plt.text(x[len(x) // 2], y[len(x) // 2] * 2, "Shoulder: " + str(shoulder_length))
        #     # plt.text(x[len(x) // 2], y[len(x) // 2] * 2, "Shoulder: " + str(shoulder_length))
        #     plt.title(
        #         func + ", " + str(len(x)) + " точек, плечо: " + str(shoulder_length) + "\nТекущая точка: " + str(i))
        #     # plt.pause(0.05)
        #     plt.show(block=False)

    # plt.get_legend().remove()
    # ("Shoulder:", mindevshould)
    # ("Deviation:", mindeviation)
    # xspace10 = np.linspace(x[parabola_start:parabola_end][0], x[parabola_start:parabola_end][-1], 500)
    xspace = np.linspace(x[0], x[-1], 500)
    y_realDrawn = [func_lambd(xi) for xi in xspace]
    # plt.clf()
    # plt.plot(x, y, 'ro', label='Точки с шумом', marker='.')
    # if showRealFunction:
    #    plt.plot(xspace, y_realDrawn, 'g', label='Реальная функция')
    # plt.plot(xspace10,fragment_smoothed, 'b', label='Parabola')
    # plt.plot(x, y_smoothed, 'bo', marker='.', label='Сглаженные точки')
    # plt.legend(loc='upper left')
    # plt.title(func + ", " + str(len(x)) + " точек, плечо: динамическое")

    # plot1.text(x[n // 2], y[n // 2] * 2, "Shoulder: " + str(shoulder_lengthh))
    # plt.show()


    deviation = 0
    if deviationType == "Squares":
        for i in range(len(y)):
            deviation += (y_real[i] - y_smoothed[i]) ** 2
    # ("shoulder_lengthh:", shoulder_lengthh)
    # ("Error:", deviation)
    elif deviationType == "AverageSquares":
        for i in range(len(y)):
            deviation += (y_real[i] - y_smoothed[i]) ** 2
        deviation /= len(y)
        deviation = math.sqrt(deviation)
    # ("shoulder_lengthh:", shoulder_lengthh)
    # ("Error:", deviation)
    elif deviationType == "Absolute":
        for i in range(len(y)):
            deviation += abs(y_real[i] - y_smoothed[i])
    # ("shoulder_lengthh:", shoulder_lengthh)
    # ("Error:", deviation)
    if plot1 is not None:
        plot1.cla()
        # (x)
        shoulderVectorStandardized = [a * ((max(y) - min(y)) / max(shoulderVector)) - max(y) * 2 for a in
                                      shoulderVector]
        # for i in range(1,len(shoulderVectorStandardized)-1):
        # shoulderVectorStandardized[i] = (shoulderVectorStandardized[i-1]+shoulderVectorStandardized[i+1])/2
        plot1.fill_between(x, shoulderVectorStandardized, min(shoulderVectorStandardized),
                           color="gray", label="График плеч")  # ,linestyle='None', marker='.')
        plot1.scatter(x[np.argmax(shoulderVectorStandardized)], max(shoulderVectorStandardized), color="black")
        plot1.text(x[np.argmax(shoulderVectorStandardized)], max(shoulderVectorStandardized),
                   f"Max: {max(shoulderVector)}")
    title = ""



    deviationyKernel = 0
    deviationyGauss = 0
    deviationyTrees = 0
    deviationyForest = 0
    deviationyMLP = 0
    deviationyNeighbours = 0

    if comparing:
        x = np.array(x).reshape(-1, 1)
        y = np.array(y)
        yKernel = kernel(x, y)
        yGauss = gauss(x, y)
        yTrees = regTrees(x, y)
        yForest = randomForest(x, y)
        yMLP = MLP(x, y)
        yNeighbours = kNeighbours(x, y)

    # else:
    #     yKernel = y
    #     yGauss = y
    #     yTrees = y
    #     yForest = y
    #     yMLP=y
    #     yNeighbours = y
    if comparing:
        for i in range(len(y)):
            deviationyKernel += (y_real[i] - yKernel[i]) ** 2
        for i in range(len(y)):
            deviationyGauss += (y_real[i] - yGauss[i]) ** 2
        for i in range(len(y)):
            deviationyTrees += (y_real[i] - yTrees[i]) ** 2
        for i in range(len(y)):
            deviationyForest += (y_real[i] - yForest[i]) ** 2
        for i in range(len(y)):
            deviationyMLP += (y_real[i] - yMLP[i]) ** 2
        for i in range(len(y)):
            deviationyNeighbours += (y_real[i] - yNeighbours[i]) ** 2

    if comparing:
        print(f"MyFunc deviation: {deviation}")
        print(f"Kernel deviation: {deviationyKernel}")
        print(f"Gauss deviation: {deviationyGauss}")
        print(f"Trees deviation: {deviationyTrees}")
        print(f"Forest deviation: {deviationyForest}")
        print(f"MLP deviation: {deviationyMLP}")
        print(f"kNeighbours deviation: {deviationyNeighbours}")
        from tkinter import filedialog
        with open(filedialog.asksaveasfilename(initialdir="SuperComparison\Fix") + ".txt", 'w+') as f:
            f.write(f"MyFunc deviation: {deviation}\n"
                    f"Kernel deviation: {deviationyKernel}\n"
                    f"Gauss deviation: {deviationyGauss}\n"
                    f"Trees deviation: {deviationyTrees}\n"
                    f"Forest deviation: {deviationyForest}\n"
                    f"MLP deviation: {deviationyMLP}\n"
                    f"kNeighbours deviation: {deviationyNeighbours}\n\n")

    if plot1 is not None:

        if comparing:
            plot1.plot(x, yKernel, color='k', label='Kernel')
            # plt.plot(x, local(x,y), color='blue', linestyle='--',linewidth=3.0, label='Local')

            plot1.plot(x, yGauss, color='r', label='Gauss')

            # plot1.plot(x, yTrees, color='orange',  label='Regression Tree')

            plot1.plot(x, yForest, color='y', label='Random Forest')
            plot1.plot(x, yMLP, color='orange', label='Neural Network')
            plot1.plot(x, yNeighbours, color='purple', label='k Nearest Neighbours')
        plot1.plot(x, y, 'ro',alpha=0.4, label='Точки с шумом')
        if showRealFunction:
            y_realDrawn = [func_lambd(xi) for xi in xspace]
            plot1.plot(xspace, y_realDrawn, 'g', label='Реальная функция')
        # plt.plot(xspace10,fragment_smoothed, 'b', label='Parabola')
        if linePlot:
            plot1.plot(x, y_smoothed, 'b', label='Сглаженные точки', linewidth=2)
        else:
            plot1.plot(x, y_smoothed, 'bo', marker='.', label='Сглаженные точки')
        plot1.legend(loc='upper left')
        title = "\nплечо: динамическое, отклонение: " + "{:.4f}".format(
            deviation) + " (Тип отклонения: " + deviationType + ")\n сглаживание концов: " + ("Да" if smoothEdges else "Нет") + ", Учитывать погрешность концов: " + "Да" if countEdges else "Нет"
    deviationFromNoised = None
    if compareWithNoised:
        deviationFromNoised = sum([abs(y0 - y1) for y0, y1 in zip(y, y_smoothed)])
    return deviation, title, deviationFromNoised


def deviationForOneDot(shoulder_length, index, x, y, y_real, yx, yx2, x2, x3, x4, deviationType):
    shoulder_length = int(shoulder_length)
    if shoulder_length > min(index, len(x) - index - 1):
        deviation = (y[index] - y_real[index])**2 * 10 * min(index, len(x) - index - 1)
    elif shoulder_length < 1:
        deviation = float("inf")
    else:
        parabola_start = index - shoulder_length
        parabola_end = index + shoulder_length
        data = (shoulder_length, parabola_start, parabola_end, x, y, yx, yx2, x2, x3, x4)
        try:
            a0, a1, a2 = np.linalg.solve(*equations(data))
        except:
            a0, a1, a2 = np.linalg.lstsq(*equations(data))[0]
        dot_smoothed = a0 + a1 * x[index] + a2 * x[index] ** 2
        if deviationType == "Squares":
            deviation = (dot_smoothed - y_real[index]) ** 2
        # ("shoulder_lengthh:", shoulder_lengthh)
        # ("Error:", deviation)
        elif deviationType == "AverageSquares":
            deviation = (dot_smoothed - y_real[index]) ** 2
            deviation /= len(y)
            deviation = math.sqrt(deviation)
        # ("shoulder_lengthh:", shoulder_lengthh)
        # ("Error:", deviation)
        elif deviationType == "Absolute":
            deviation = abs(dot_smoothed - y_real[index])
    return deviation


def findBestFixedShoulderWithoutRealFunction(x, y, y_real, yx, yx2, x2, x3, x4, func_lambd, smoothEdges=True,
                                             deviationType="Squares", countEdges=True, realtime=False,
                                             progress=None, window=None, savepath=None, table=None, alg=None,
                                             dichotomyMin=None, dichotomyMax=None, compareWithNoised=False,
                                             shoulderLimit=None, showGraphs=True, plot21=None, plot22=None):
    shoulderLimit = min(shoulderLimit, (len(x) - 1) // 2)
    global working
    bestShoulder = 6
    minDeviation = float("inf")
    count = 0
    deviations = []
    deviationsFromNoised = []
    shoulders = []
    derivatives = []
    secondDerivatives = []
    fig, axs = plt.subplots(3)
    if shoulderLimit is None:
        shoulderLimit = len(x) // 2 - 1
    if alg == "Loop":
        for shoulder in range(1, shoulderLimit + 1):
            if working:
                if progress is not None:
                    progress.config(value=100 * shoulder / shoulderLimit)
                    if window is not None:
                        window.update()
                shoulders.append(shoulder)
                _, deviation, title, deviationFromNoised = smooth(x, y, y_real, yx, yx2, x2, x3, x4, func_lambd,
                                                                  table=table, shoulder_lengthh=shoulder,
                                                                  smoothEdges=smoothEdges,
                                                                  deviationType=deviationType, countEdges=countEdges,
                                                                  savepath=savepath,
                                                                  compareWithNoised=compareWithNoised)
                # deviations.append(deviation)
                # if deviationsFromNoised:
                #     derivative = deviationFromNoised - deviationsFromNoised[-1]
                #     derivatives.append(derivative)
                deviationsFromNoised.append(deviationFromNoised)

                # if len(deviationsFromNoised) >= 3:
                #     secondDerivatives.append(
                #         deviationsFromNoised[-1] - 2 * deviationsFromNoised[-2] + deviationsFromNoised[-3])

                # count += 1
                # if deviation < minDeviation:
                #     count = 0
                #     minDeviation = deviation
                #     bestShoulder = shoulder
                # if count == 30:
                # break
                # if realtime:
                #     axs[0].cla()
                #     axs[0].set_title("Отклонение от реальных точек")
                #     axs[0].plot(shoulders, deviations, 'g', label="Зависимость ошибки от плеча")
                #     axs[0].scatter(bestShoulder, minDeviation, s=30)
                #     axs[0].text(bestShoulder, minDeviation, "Оптимальное плечо:" + str(bestShoulder))
                #
                #     axs[1].cla()
                #     axs[1].set_title("Отклонение от зашумлённых точек")
                #     axs[1].plot(shoulders, deviationsFromNoised, 'r', label="Отклонение от зашумлённых точек")
                #     axs[1].scatter(bestShoulder, deviationsFromNoised[bestShoulder - 1], s=30)
                #     axs[1].text(bestShoulder, deviationsFromNoised[bestShoulder - 1],
                #                 "Оптимальное плечо:" + str(bestShoulder))

                # plt.pause(0.0001)
                # fig.show()

        # smoothedDeviationsFromNoised = [deviationsFromNoised[0]]+[(deviationsFromNoised[i-1] + deviationsFromNoised[i+1])/2 for i in range(1, len(deviationsFromNoised)-1)]+[deviationsFromNoised[-1]]
        # deviationsFromNoised = smoothedDeviationsFromNoised
        # print("devNlen",len(deviationsFromNoised))

        derivatives = [deviationsFromNoised[i] - deviationsFromNoised[i - 1] for i in
                       range(1, len(deviationsFromNoised))]
        startFlat = None
        endFlat = shoulderLimit - 1
        # endFlat = None
        for i in range(len(derivatives)):
            if not startFlat:
                if sum(derivatives[i - 1:i + 2]) / 3 < 2:
                    startFlat = i
            elif sum(derivatives[i - 1:i + 2]) / 3 > 2 or i + 1 > len(derivatives):
                endFlat = i
                break
        bestShoulder = (endFlat + startFlat) // 2

        # smoothedDerivatives = [derivatives[0]]+[(derivatives[i-1] + derivatives[i+1])/2 for i in range(1, len(derivatives)-1)]+[derivatives[-1]]
        # derivatives = smoothedDerivatives

        # secondDerivatives = [derivatives[i] - derivatives[i-1] for i in range(1, len(derivatives))]
        # # smoothedSecondDerivatives = [secondDerivatives[0]]+[(secondDerivatives[i-1] + secondDerivatives[i+1])/2 for i in range(1, len(secondDerivatives)-1)]+[secondDerivatives[-1]]
        # # secondDerivatives = smoothedSecondDerivatives
        #
        # minDeltaY = derivatives[3]
        # bestShoulder = 3
        # # mean = sum(derivatives[3:])/len(derivatives[3:])
        # # for i,dot in enumerate(derivatives):
        # #     #print(f"{abs(dot-minDeltaY)} >= {mean*0.1}")
        # #     print("best",bestShoulder)
        # #     if dot < minDeltaY and abs(dot-minDeltaY) > mean*0.1:
        # #         print("bop")
        # #         minDeltaY = dot
        # #         bestShoulder = i+1
        #
        # # bestShoulder = derivatives.index(min(derivatives)) + 1
        #
        # smoothedSecondDerivativeAbs = [abs(e) for e in secondDerivatives]
        # smoothedSecondDerivativeAbs = [smoothedSecondDerivativeAbs[0], smoothedSecondDerivativeAbs[0]] + smoothedSecondDerivativeAbs
        # #secondDerivatives = [secondDerivatives[0], secondDerivatives[0]] + secondDerivatives
        #
        # #derivatives.insert(0, derivatives[0])
        #
        # # derivatives = [derivatives[0]] + derivatives.tolist()
        # # secondDerivatives = [secondDerivatives[0], secondDerivatives[0]] + secondDerivatives.tolist()
        # # print("min",min(smoothedSecondDerivativeAbs))
        # # minn = float("inf")
        # # for i in range(1,len(smoothedSecondDerivativeAbs)-1):
        # #     if smoothedSecondDerivativeAbs[i] < minn and secondDerivatives[i-1] < 0 and secondDerivatives[i-1] < secondDerivatives[i] and secondDerivatives[i+1] > secondDerivatives[i]  and secondDerivatives[i+1] > 0:
        # #         minn = smoothedSecondDerivativeAbs[i]
        # #         bestShoulder = i + 1
        # #        # minn = dev
        # #        # bestShoulder = i + 1
        # # print("best", bestShoulder)
        # # print(smoothedSecondDerivativeAbs)
        # # bestShoulder = secondDerivatives.index(min(secondDerivatives)) + 1
        # # bestShoulder = secondDerivatives.index(min(secondDerivatives)) + 1
        # # bestShoulder = smoothedSecondDerivativeAbs.index(min(smoothedSecondDerivativeAbs)) + 1
        # # print("best", bestShoulder)
        # # bestShoulder = secondDerivatives.index(min(secondDerivatives)) + 1
        # # for i,e in enumerate(secondDerivatives):
        # #     if e > 0:
        # #         firstPositive = i
        # #         break
        # # firstPositive*=2
        # # for i in range(firstPositive,len(secondDerivatives)):
        # #     print(f"{sum(secondDerivatives[firstPositive:i])} < {abs(sum(secondDerivatives[i:2*i-firstPositive]))}")
        # #     if abs(sum(secondDerivatives[firstPositive:i])) < abs(sum(secondDerivatives[i:2*i-firstPositive])):
        # #         print("FFFF")
        # #         bestShoulder = i +1
        # #         break
        #
        # # for i in range(4, len(secondDerivatives)):
        # #     print(f"{sum(secondDerivatives[i-10:i])} < {abs(sum(secondDerivatives[i:2 * i - 10]))}")
        # #     if abs(sum(secondDerivatives[i-10:i])) < abs(sum(secondDerivatives[i:2 * i - 10])):
        # #         print("FFFF")
        # #         bestShoulder = i + 1
        # #         break
        # # bestShoulder = secondDerivatives.index(min(secondDerivatives)) + 1
        #
        # firstDerivativeLessThan1 = 5
        # flatEnd = 10
        # for i, e in enumerate(derivatives):
        #     if e < 1:
        #         firstDerivativeLessThan1 = i
        #         break
        # for i in range (firstDerivativeLessThan1+3,len(derivatives)):
        #     if sum(derivatives[i-3:i+4])/7 > 0.45:
        #         flatEnd = i
        #         break
        # print(f"FlatEnd:{flatEnd}")
        # cubic_coeffs = np.polyfit(shoulders[:flatEnd], deviationsFromNoised[:flatEnd], 3)
        # second_derivative_coeffs = np.polyder(cubic_coeffs, m=2)
        # roots = np.roots(second_derivative_coeffs)
        # print(f"Roots: {roots}")
        # bestShoulder = math.floor(roots[0])
    # elif alg == "Dichotomy":
    #     log2length = float(log(dichotomyMax - dichotomyMin, 2))
    #     print(log2length)
    #     eps = 3.1
    #     delta = 1
    #     a = dichotomyMin
    #     b = dichotomyMax
    #     while abs(b - a) >= eps:
    #         progress.config(value=(log2length - float(log(abs(b - a), 2)) + 1) * 100 / log2length)
    #         window.update()
    #
    #         print(f"a: {a}, b: {b}")
    #         lamb = (a + b) // 2 - delta
    #         mu = (a + b) // 2 + delta
    #         print(f"lamb: {lamb}, mu: {mu}")
    #
    #         _, flamb, _, _ = smooth(x, y, y_real, yx, yx2, x2, x3, x4, func_lambd, table=table,
    #                                 shoulder_lengthh=lamb, smoothEdges=smoothEdges,
    #                                 deviationType=deviationType, countEdges=countEdges, savepath=savepath)
    #         _, fmu, _, _ = smooth(x, y, y_real, yx, yx2, x2, x3, x4, func_lambd, table=table,
    #                               shoulder_lengthh=mu, smoothEdges=smoothEdges,
    #                               deviationType=deviationType, countEdges=countEdges, savepath=savepath)
    #         if flamb < fmu:
    #             b = mu
    #         else:
    #             a = lamb
    #     bestShoulder = (a + b) // 2

    # fig.suptitle('Vertically stacked subplots')
    # axs[0].plot(x, y)
    # axs[1].plot(x, -y)

    # plt.cla()
    # axs[0].set_title("Отклонение от реальных точек")
    # axs[0].plot(shoulders, deviations, 'g', label="Зависимость ошибки от плеча")
    # axs[0].scatter(bestShoulder, minDeviation, s=30)
    # axs[0].text(bestShoulder, minDeviation, "Оптимальное плечо:" + str(bestShoulder))

    if progress is not None:
        progress.config(value=100)

    if compareWithNoised:
        # plt.cla()
        axs[0].set_title("Грфик производных")
        axs[0].plot(shoulders[1:], derivatives, 'b')
        axs[0].scatter(bestShoulder, derivatives[bestShoulder - 1], s=30)
        axs[0].text(bestShoulder, derivatives[bestShoulder - 1], "Оптимальное плечо:" + str(bestShoulder))

        axs[1].set_title("Отклонение от зашумлённых точек")
        axs[1].plot(shoulders, deviationsFromNoised, 'r', label="Отклонение от зашумлённых точек")
        axs[1].scatter(bestShoulder, deviationsFromNoised[bestShoulder - 1], s=30)
        axs[1].text(bestShoulder, deviationsFromNoised[bestShoulder - 1], "Оптимальное плечо:" + str(bestShoulder))

        print(startFlat)
        axs[1].scatter(startFlat, deviationsFromNoised[startFlat], c="orange", marker="|", s=200)
        axs[1].scatter(endFlat, deviationsFromNoised[endFlat], c="orange", marker="|", s=200)

        # axs[2].set_title("Вторая производная")
        # axs[2].plot(shoulders[2:], secondDerivatives, 'black', label="Отклонение от зашумлённых точек")
        # axs[2].scatter(bestShoulder, secondDerivatives[bestShoulder - 1], s=30)
        # axs[2].text(bestShoulder, secondDerivatives[bestShoulder - 1], "Оптимальное плечо:" + str(bestShoulder))

        plot21.set_title("Грфик производных")
        plot21.plot(shoulders[1:], derivatives, 'b')
        plot21.scatter(bestShoulder, derivatives[bestShoulder - 1], s=30)
        plot21.text(bestShoulder, derivatives[bestShoulder - 1], str(bestShoulder))

        plot22.set_title("Отклонение от зашумлённых точек")
        plot22.plot(shoulders, deviationsFromNoised, 'r', label="Отклонение от зашумлённых точек")
        plot22.scatter(bestShoulder, deviationsFromNoised[bestShoulder - 1], s=30)
        plot22.text(bestShoulder, deviationsFromNoised[bestShoulder - 1], str(bestShoulder))

        # axs[1].show(block=False)
        # axs[1].savefig(savepath + "/DistanceFromNoised.png")
    if showGraphs: fig.show()
    if savepath is not None and savepath != "" and alg == "Loop":
        savepath = savepath.replace("\n", "")
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        fig.savefig(savepath + "/ShouldersErrors.png")
    return bestShoulder
