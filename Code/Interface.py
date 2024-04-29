import configparser
import shutil
import time
import tkinter as tk
import warnings
from tkinter import filedialog
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import Approximator
from Approximator import *
from Generator import *

warnings.filterwarnings('ignore')

shoulder_length = 0
parabola_start = 0
parabola_end = 0
x = []
y = []
yx = []
yx2 = []
x2 = []
x3 = []
x4 = []
Approximator.working = False


# Create a textbox that only accepts integer values
def loadDots(path):
    shutil.copy(path, "../dots.xlsx")


def loadSettings(path=None):
    if path:
        shutil.copy(path, "../settings.ini")
    config = configparser.ConfigParser()
    config.read("../settings.ini")
    shoulderVar.set(config["Approximator"]["shoulderLen"])
    iterationsVar.set(config["Approximator"]["maxIterations"])
    accuracyVar.set(config["Approximator"]["accuracy"])
    populationSizeVar.set(config["Approximator"]["populationSize"])
    mutationProbVar.set(config["Approximator"]["mutatuionProb"])
    smoothVar.set(config["Approximator"]["smoothEdges"])

    showRealFuncVar.set(config["Approximator"]["showRealFunc"])

    сountEdgesVar.set(config["Approximator"]["countEdges"])

    realTimeVar.set(config["Approximator"]["realTime"])

    savePlotsVar.set(config["Approximator"]["savePlots"])



    compareWithNoisedVar.set(config["Approximator"]["compareWithNoised"])

    generateWholeVectorVar.set(config["Approximator"]["generateWholeVector"])
    deviationTypeVar.set(config["Approximator"]["deviation"])
    if config["Approximator"]["deviation"] == "Squares":
        radioSquares.select()
    elif config["Approximator"]["deviation"] == "Absolute":
        radioAbsolute.select()
    elif config["Approximator"]["deviation"] == "AverageSquares":
        radioAverageSquares.select()
    shoulderTypeVar.set(config["Approximator"]["shoulderType"])
    if config["Approximator"]["shoulderType"] == "Fixed":
        radioFixedShoulder.select()
        toggleFrame(optimizationFrame, 0)
        toggleFrame(fixedOptimalFrame, 0)
    elif config["Approximator"]["shoulderType"] == "FixedOptimal":
        radioFixedOptimal.select()
        shoulderEntry.config(state="disabled")
        print("toggling!!!")
        toggleFrame(optimizationFrame, 0)
    elif config["Approximator"]["shoulderType"] == "SeparateOptimal":
        radioSeparateOptimal.select()
        shoulderEntry.config(state="disabled")
        toggleFrame(fixedOptimalFrame, 0)
    optimizationMethodVar.set(config["Approximator"]["optimizationMethod"])
    if config["Approximator"]["optimizationMethod"] == "Loop":
        radioLoop.select()
        toggleFrame(geneticFrame, 0)
    elif config["Approximator"]["optimizationMethod"] == "Random":
        radioRandom.select()
        toggleFrame(geneticFrame, 0)
    elif config["Approximator"]["optimizationMethod"] == "Dichotomy":
        radioDynamicDichotomy.select()
        toggleFrame(geneticFrame, 0)
    elif config["Approximator"]["optimizationMethod"] == "Genetic":
        radioGenetic.select()
    isShoulderLimitVar.set(config["Approximator"]["isShoulderLimit"])
    if config["Approximator"]["isShoulderLimit"] != "1":
        shoulderLimitEntry.config(state="disabled")
    shoulderLimitVar.set(config["Approximator"]["shoulderLimit"])
    fixedOptimalAlgVar.set(config["Approximator"]["fixedOptimalAlg"])
    if fixedOptimalAlgVar.get() != "Dichotomy":
        toggleFrame(dichotomyFrame, 0)
    minDichotomyVar.set(config["Approximator"]["dichotomyMin"])
    maxDichotomyVar.set(config["Approximator"]["dichotomyMax"])
    saveDirectoryVar.set(config["Approximator"]["saveDirectory"])
    noRealFunctionVar.set(config["Approximator"]["noRealFunction"])
    sideShoulderTypeVar.set(config["Approximator"]["sideShoulderType"])
    comparingVar.set(config["Approximator"]["comparing"])
    minDichotomyVar.set(config["Approximator"]["dichotomymin"])
    maxDichotomyVar.set(config["Approximator"]["dichotomymax"])
    minGoldenVar.set(config["Approximator"]["minGolden"])
    maxGoldenVar.set(config["Approximator"]["maxGolden"])
    amortizeSidesVar.set(config["Approximator"]["amortizeSides"])
    sideLengthVar.set(config["Approximator"]["sideLength"])

    functionVar.set(config["Generator"]["func"])
    intervalMinVar.set(config["Generator"]["intervalMin"])
    intervalMaxVar.set(config["Generator"]["intervalMax"])
    dotsCountVar.set(config["Generator"]["dotsCount"])
    noiseMinVar.set(config["Generator"]["minNoise"])
    noiseMaxVar.set(config["Generator"]["maxNoise"])
    normalMeanVar.set(config["Generator"]["normalDistMean"])
    normalDispersionVar.set(config["Generator"]["normalDistDispersion"])
    superNoiseMinVar.set(config["Generator"]["superNoiseMin"])
    superNoiseMaxVar.set(config["Generator"]["superNoiseMax"])
    superNoisePercentVar.set(config["Generator"]["superNoisesPercent"])
    minStepVar.set(config["Generator"]["randomStepMin"])
    maxStepVar.set(config["Generator"]["randomStepMax"])
    segmentCountVar.set(config["Generator"]["segmentsCount"])
    percentToFillSegmentVar.set(config["Generator"]["percentageToFillSegment"])
    minPercentToFillEmptyVar.set(config["Generator"]["minPercentageToFillEmpty"])
    maxPercentToFillEmptyVar.set(config["Generator"]["maxPercentageToFillEmpty"])
    randomStepVar.set(config["Generator"]["isRandomStep"])
    modeVar.set(config["Generator"]["mode"])
    if config["Generator"]["mode"] == "1":
        radioFixedStep.select()
        toggleFrame(randomStepFrame, 0)
        toggleFrame(segmentedFrame, 0)
    elif config["Generator"]["mode"] == "2":
        radioRandom.select()
        toggleFrame(segmentedFrame, 0)
    elif config["Generator"]["mode"] == "3":
        radioSegmented.select()
        toggleFrame(randomStepFrame, 0)

    noiseVar.set(config["Generator"]["noise"])
    if config["Generator"]["noise"] == "1":
        checkboxNoise.select()
    else:
        toggleFrame(noisesFrame, 0)
    noisesRelativeVar.set(config["Generator"]["noiseRelative"])
    if config["Generator"]["noiseRelative"] == 1:
        checkboxNoisesRelative.select()

    distVar.set(config["Generator"]["distribution"])
    if config["Generator"]["distribution"] == "normal":
        radioNormal.select()
        toggleFrame(uniformDistFrame, 0)
    elif config["Generator"]["distribution"] == "uniform":
        radioNeNormal.select()
        toggleFrame(normalFrame, 0)

    superNoiseVar.set(config["Generator"]["superNoises"])
    if config["Generator"]["superNoises"] == "1":
        checkboxNoise.select()
    else:
        toggleFrame(superNoisesFrame, 0)
    superNoisesRelativeVar.set(config["Generator"]["superNoiseRelative"])
    if config["Generator"]["superNoiseRelative"] == 1:
        checkboxSuperNoisesRelative.select()


def saveApproximatorSettings(path=None):
    config = configparser.ConfigParser()
    config.read("../settings.ini")
    #     config["Approximator"]["shoulderLen"] = str(shoulderVar.get())
    #     config["Approximator"]["smoothEdges"] = str(smoothVar.get())
    #     config["Approximator"]["showRealFunc"] = str(showRealFuncVar.get())
    #     config["Approximator"]["countEdges"] = str(сountEdgesVar.get())
    #     config["Approximator"]["deviation"] = deviationTypeVar.get()
    #     config["Approximator"]["realTime"] = str(realTimeVar.get())
    #     config["Approximator"]["savePlots"] = str(savePlotsVar.get())
    #     config["Approximator"]["compareWithNoised"] = str(compareWithNoisedVar.get())
    #     config["Approximator"]["shoulderType"] = str(shoulderTypeVar.get())
    #     config["Approximator"]["generateWholeVector"] = str(generateWholeVectorVar.get())
    #     config["Approximator"]["maxIterations"] = str(iterationsVar.get())
    #     config["Approximator"]["accuracy"] = str(accuracyVar.get())
    #     config["Approximator"]["optimizationMethod"] = str(optimizationMethodVar.get())
    #     config["Approximator"]["populationSize"] = str(populationSizeVar.get())
    #     config["Approximator"]["mutatuionProb"] = str(mutationProbVar.get())
    #     config["Approximator"]["isShoulderLimit"] = str(isShoulderLimitVar.get())
    #     config["Approximator"]["shoulderLimit"] = str(shoulderLimitVar.get())
    #     config["Approximator"]["isShoulderLimit"] = str(isShoulderLimitVar.get())
    # #    config["Approximator"]["saveDirectory"] = str(saveDirectoryVar.get()).replace("⁄", "/")
    #     config["Approximator"]["noRealFunction"] = str(noRealFunctionVar.get())
    config["Approximator"]["shoulderLen"] = str(shoulderVar.get())
    config["Approximator"]["smoothEdges"] = str(smoothVar.get())
    config["Approximator"]["showRealFunc"] = str(showRealFuncVar.get())
    config["Approximator"]["countEdges"] = str(сountEdgesVar.get())
    config["Approximator"]["deviation"] = deviationTypeVar.get()
    config["Approximator"]["realTime"] = str(realTimeVar.get())
    config["Approximator"]["savePlots"] = str(savePlotsVar.get())
    config["Approximator"]["compareWithNoised"] = str(compareWithNoisedVar.get())
    config["Approximator"]["shoulderType"] = str(shoulderTypeVar.get())
    config["Approximator"]["generateWholeVector"] = str(generateWholeVectorVar.get())
    config["Approximator"]["maxIterations"] = str(iterationsVar.get())
    config["Approximator"]["accuracy"] = str(accuracyVar.get())
    config["Approximator"]["optimizationMethod"] = str(optimizationMethodVar.get())
    config["Approximator"]["populationSize"] = str(populationSizeVar.get())
    config["Approximator"]["mutatuionProb"] = str(mutationProbVar.get())
    config["Approximator"]["isShoulderLimit"] = str(isShoulderLimitVar.get())
    config["Approximator"]["shoulderLimit"] = str(shoulderLimitVar.get())
    config["Approximator"]["isShoulderLimit"] = str(isShoulderLimitVar.get())
    config["Approximator"]["noRealFunction"] = str(noRealFunctionVar.get())
    config["Approximator"]["saveDirectory"] = str(saveDirectoryVar.get())
    config["Approximator"]["sideShoulderType"] = str(sideShoulderTypeVar.get())
    print(comparingVar.get())
    config["Approximator"]["comparing"] = str(comparingVar.get())
    config["Approximator"]["dichotomymin"] = str(minDichotomyVar.get())
    config["Approximator"]["dichotomymax"] = str(maxDichotomyVar.get())
    config["Approximator"]["minGolden"] = str(minGoldenVar.get())
    config["Approximator"]["maxGolden"] = str(maxGoldenVar.get())
    config["Approximator"]["amortizeSides"] = str(amortizeSidesVar.get())
    config["Approximator"]["sideLength"] = str(sideLengthVar.get())

    if path:
        if path[-4:] != ".ini": path += ".ini"
        shutil.copy("../settings.ini", path)
    else:
        path = "../settings.ini"
    with open(path, 'w') as configfile:
        config.write(configfile)


def saveGeneratorSettings(path=None):
    config = configparser.ConfigParser()
    config.read("../settings.ini")
    # config["Generator"]["func"] = str(functionVar.get())
    # config["Generator"]["intervalMin"] = str(intervalMinVar.get())
    # config["Generator"]["intervalMax"] = str(intervalMaxVar.get())
    # config["Generator"]["dotsCount"] = str(dotsCountVar.get())
    # config["Generator"]["randomStepMin"] = str(minStepVar.get())
    # config["Generator"]["randomStepMax"] = str(maxStepVar.get())
    # config["Generator"]["segmentsCount"] = str(segmentCountVar.get())
    # config["Generator"]["percentageToFillSegment"] = str(percentToFillSegmentVar.get())
    # config["Generator"]["mode"] = str(modeVar.get())
    # config["Generator"]["noise"] = str(noiseVar.get())
    # config["Generator"]["noiseRelative"] = str(noisesRelativeVar.get())
    # config["Generator"]["minNoise"] = str(noiseMinVar.get())
    # config["Generator"]["maxNoise"] = str(noiseMaxVar.get())
    # config["Generator"]["distribution"] = str(distVar.get())
    # config["Generator"]["normalDistMean"] = str(normalMeanVar.get())
    # config["Generator"]["normalDistDispersion"] = str(normalDispersionVar.get())
    # config["Generator"]["superNoises"] = str(superNoiseVar.get())
    # config["Generator"]["superNoiseRelative"] = str(superNoisesRelativeVar.get())
    # config["Generator"]["superNoiseMin"] = str(superNoiseMinVar.get())
    # config["Generator"]["superNoiseMax"] = str(superNoiseMaxVar.get())
    # config["Generator"]["superNoisesPercent"] = str(superNoisePercentVar.get())
    config["Generator"]["func"] = str(functionVar.get())
    config["Generator"]["intervalMin"] = str(intervalMinVar.get())
    config["Generator"]["intervalMax"] = str(intervalMaxVar.get())
    config["Generator"]["dotsCount"] = str(dotsCountVar.get())
    config["Generator"]["randomStepMin"] = str(minStepVar.get())
    config["Generator"]["randomStepMax"] = str(maxStepVar.get())
    config["Generator"]["segmentsCount"] = str(segmentCountVar.get())
    config["Generator"]["percentageToFillSegment"] = str(percentToFillSegmentVar.get())
    config["Generator"]["minPercentageToFillEmpty"] = str(minPercentToFillEmptyVar.get())
    config["Generator"]["maxPercentageToFillEmpty"] = str(maxPercentToFillEmptyVar.get())
    config["Generator"]["isRandomStep"] = str(randomStepVar.get())
    config["Generator"]["mode"] = str(modeVar.get())
    config["Generator"]["noise"] = str(noiseVar.get())
    config["Generator"]["noiseRelative"] = str(noisesRelativeVar.get())
    config["Generator"]["minNoise"] = str(noiseMinVar.get())
    config["Generator"]["maxNoise"] = str(noiseMaxVar.get())
    config["Generator"]["distribution"] = str(distVar.get())
    config["Generator"]["normalDistMean"] = str(normalMeanVar.get())
    config["Generator"]["normalDistDispersion"] = str(normalDispersionVar.get())
    config["Generator"]["superNoises"] = str(superNoiseVar.get())
    config["Generator"]["superNoiseRelative"] = str(superNoisesRelativeVar.get())
    config["Generator"]["superNoiseMin"] = str(superNoiseMinVar.get())
    config["Generator"]["superNoiseMax"] = str(superNoiseMaxVar.get())
    config["Generator"]["superNoisesPercent"] = str(superNoisePercentVar.get())
    if path:
        if path[-4:] != ".ini": path += ".ini"
        shutil.copy("../settings.ini", path)
    else:
        path = "../settings.ini"
    with open(path, 'w') as configfile:
        config.write(configfile)


def saveAllSettings(path=None):
    config = configparser.ConfigParser()
    config.read("../settings.ini")
    config["Approximator"]["shoulderLen"] = str(shoulderVar.get())
    config["Approximator"]["smoothEdges"] = str(smoothVar.get())
    config["Approximator"]["showRealFunc"] = str(showRealFuncVar.get())
    config["Approximator"]["countEdges"] = str(сountEdgesVar.get())
    config["Approximator"]["deviation"] = deviationTypeVar.get()
    config["Approximator"]["realTime"] = str(realTimeVar.get())
    config["Approximator"]["savePlots"] = str(savePlotsVar.get())
    config["Approximator"]["compareWithNoised"] = str(compareWithNoisedVar.get())
    config["Approximator"]["shoulderType"] = str(shoulderTypeVar.get())
    config["Approximator"]["generateWholeVector"] = str(generateWholeVectorVar.get())
    config["Approximator"]["maxIterations"] = str(iterationsVar.get())
    config["Approximator"]["accuracy"] = str(accuracyVar.get())
    config["Approximator"]["optimizationMethod"] = str(optimizationMethodVar.get())
    config["Approximator"]["populationSize"] = str(populationSizeVar.get())
    config["Approximator"]["mutatuionProb"] = str(mutationProbVar.get())
    config["Approximator"]["isShoulderLimit"] = str(isShoulderLimitVar.get())
    config["Approximator"]["shoulderLimit"] = str(shoulderLimitVar.get())
    config["Approximator"]["isShoulderLimit"] = str(isShoulderLimitVar.get())
    config["Approximator"]["noRealFunction"] = str(noRealFunctionVar.get())
    config["Approximator"]["saveDirectory"] = str(saveDirectoryVar.get())
    config["Approximator"]["sideShoulderType"] = str(sideShoulderTypeVar.get())
    config["Approximator"]["comparing"] = str(comparingVar.get())
    config["Approximator"]["dichotomymin"] = str(minDichotomyVar.get())
    config["Approximator"]["dichotomymax"] = str(maxDichotomyVar.get())
    config["Approximator"]["minGolden"] = str(minGoldenVar.get())
    config["Approximator"]["maxGolden"] = str(maxGoldenVar.get())
    config["Approximator"]["amortizeSides"] = str(amortizeSidesVar.get())
    config["Approximator"]["sideLength"] = str(sideLengthVar.get())

    config["Generator"]["func"] = str(functionVar.get())
    config["Generator"]["intervalMin"] = str(intervalMinVar.get())
    config["Generator"]["intervalMax"] = str(intervalMaxVar.get())
    config["Generator"]["dotsCount"] = str(dotsCountVar.get())
    config["Generator"]["randomStepMin"] = str(minStepVar.get())
    config["Generator"]["randomStepMax"] = str(maxStepVar.get())
    config["Generator"]["segmentsCount"] = str(segmentCountVar.get())
    config["Generator"]["percentageToFillSegment"] = str(percentToFillSegmentVar.get())
    config["Generator"]["minPercentageToFillEmpty"] = str(minPercentToFillEmptyVar.get())
    config["Generator"]["maxPercentageToFillEmpty"] = str(maxPercentToFillEmptyVar.get())
    config["Generator"]["isRandomStep"] = str(randomStepVar.get())
    config["Generator"]["mode"] = str(modeVar.get())
    config["Generator"]["noise"] = str(noiseVar.get())
    config["Generator"]["noiseRelative"] = str(noisesRelativeVar.get())
    config["Generator"]["minNoise"] = str(noiseMinVar.get())
    config["Generator"]["maxNoise"] = str(noiseMaxVar.get())
    config["Generator"]["distribution"] = str(distVar.get())
    config["Generator"]["normalDistMean"] = str(normalMeanVar.get())
    config["Generator"]["normalDistDispersion"] = str(normalDispersionVar.get())
    config["Generator"]["superNoises"] = str(superNoiseVar.get())
    config["Generator"]["superNoiseRelative"] = str(superNoisesRelativeVar.get())
    config["Generator"]["superNoiseMin"] = str(superNoiseMinVar.get())
    config["Generator"]["superNoiseMax"] = str(superNoiseMaxVar.get())
    config["Generator"]["superNoisesPercent"] = str(superNoisePercentVar.get())
    if path:
        if path[-4:] != ".ini": path += ".ini"
        shutil.copy("../settings.ini", path)
    else:
        path = "../settings.ini"

    with open(path, 'w') as configfile:
        config.write(configfile)


def validateInt(value):
    if value.isdigit() or value == "":
        return True
    return False


def validateFloat(value):
    # Check if the new value is a valid float
    if value == "" or value == "-":
        return True
    try:
        float(value)
        return True
    except ValueError:
        return False


def toggleFrame(frame, value):
    if value == 0:
        for child in frame.winfo_children():
            try:
                child.configure(state='disable')
            except:
                toggleFrame(child, 0)
    elif value == 1:
        for child in frame.winfo_children():
            try:
                child.configure(state='normal')
            except:
                toggleFrame(child, 1)


def on_stop_button_click():
    Approximator.working = False
    print("on")


def on_button_click1():
    showGraphs = True
    saveApproximatorSettings()
    Approximator.working = True
    beginTime = time.time()
    df = pd.read_excel('../dots.xlsx')
    symbx = symbols('x')
    func = df.columns[0].split('=')[1]
    func_lambd = lambdify(symbx, func)
    x = df['x'].tolist()
    y = df['f(x)'].tolist()
    y_real = df['f(x) Real'].tolist()
    n = len(x)
    yx = [xi * yi for xi, yi in zip(x, y)]
    yx2 = [xi ** 2 * yi for xi, yi in zip(x, y)]
    x2 = [xi ** 2 for xi in x]
    x3 = [xi ** 3 for xi in x]
    x4 = [xi ** 4 for xi in x]
    plot21.cla()
    plot22.cla()
    mpl.pyplot.cla()
    savepath = saveDirectoryVar.get()
    print(savepath)

    if shoulderTypeVar.get() == "Fixed":
        plt, deviation, title, _ = smooth(x, y, y_real, yx, yx2, x2, x3, x4, func_lambd,
                                          sideShoulderType=sideShoulderTypeVar.get(), comparing=comparingVar.get(),
                                          savepath=savepath, compareWithNoised=compareWithNoisedVar.get(),
                                          func=func, table=table, plot1=plot1,
                                          shoulder_lengthh=int(shoulderVar.get()), smoothEdges=smoothVar.get(),
                                          showRealFunction=showRealFuncVar.get(), countEdges=сountEdgesVar.get(),
                                          deviationType=deviationTypeVar.get(), realtime=realTimeVar.get(),linePlot=linePlotVar.get())
    elif shoulderTypeVar.get() == "FixedOptimal":
        if noRealFunctionVar.get():
            # print("Всё оК!!!!!")
            plt, deviation, title, _ = smooth(x, y, y_real, yx, yx2, x2, x3, x4, func_lambd,
                                              compareWithNoised=compareWithNoisedVar.get(),
                                              comparing=comparingVar.get(), func=func, table=table, plot1=plot1,
                                              shoulder_lengthh=findBestFixedShoulderWithoutRealFunction(x, y,
                                                                                                        y_real, yx,
                                                                                                        yx2, x2, x3,
                                                                                                        x4,
                                                                                                        func_lambd,
                                                                                                        plot21=plot21,
                                                                                                        plot22=plot22,
                                                                                                        showGraphs=showGraphs,
                                                                                                        compareWithNoised=compareWithNoisedVar.get(),
                                                                                                        dichotomyMin=int(
                                                                                                            minDichotomyVar.get()),
                                                                                                        dichotomyMax=int(
                                                                                                            maxDichotomyVar.get()),
                                                                                                        alg=fixedOptimalAlgVar.get(),
                                                                                                        savepath=savepath,
                                                                                                        table=table,
                                                                                                        smoothEdges=smoothVar.get(),
                                                                                                        deviationType=deviationTypeVar.get(),
                                                                                                        countEdges=сountEdgesVar.get(),
                                                                                                        realtime=realTimeVar.get(),
                                                                                                        progress=progress,
                                                                                                        window=window,
                                                                                                        shoulderLimit=int(
                                                                                                            shoulderLimitVar.get())),
                                              smoothEdges=smoothVar.get(), showRealFunction=showRealFuncVar.get(),
                                              countEdges=сountEdgesVar.get(),
                                              deviationType=deviationTypeVar.get(),linePlot=linePlotVar.get())  # realTimeVar.get())
        else:
            plt, deviation, title, _ = smooth(x, y, y_real, yx, yx2, x2, x3, x4, func_lambd,
                                              sideShoulderType=sideShoulderTypeVar.get(),
                                              compareWithNoised=compareWithNoisedVar.get(),
                                              comparing=comparingVar.get(), func=func, table=table, plot1=plot1,
                                              shoulder_lengthh=findBestShoulder(x, y, y_real, yx, yx2, x2, x3, x4,
                                                                                func_lambd,
                                                                                goldenMin=minGoldenVar.get(),
                                                                                goldenMax=maxGoldenVar.get(),
                                                                                tau=tauGoldenVar.get(),
                                                                                sideShoulderType=sideShoulderTypeVar.get(),
                                                                                showGraphs=showGraphs,
                                                                                plot21=plot21, plot22=plot22,
                                                                                compareWithNoised=compareWithNoisedVar.get(),
                                                                                dichotomyMin=int(
                                                                                    minDichotomyVar.get()),
                                                                                dichotomyMax=int(
                                                                                    maxDichotomyVar.get()),
                                                                                alg=fixedOptimalAlgVar.get(),
                                                                                savepath=savepath, table=table,
                                                                                smoothEdges=smoothVar.get(),
                                                                                deviationType=deviationTypeVar.get(),
                                                                                countEdges=сountEdgesVar.get(),
                                                                                realtime=realTimeVar.get(),
                                                                                progress=progress, window=window,
                                                                                shoulderLimit=int(
                                                                                    shoulderLimitVar.get())),
                                              smoothEdges=smoothVar.get(), showRealFunction=showRealFuncVar.get(),
                                              countEdges=сountEdgesVar.get(), deviationType=deviationTypeVar.get(),
                                              savepath=savepath,linePlot=linePlotVar.get())  # realTimeVar.get())
    elif shoulderTypeVar.get() == "SeparateOptimal":
        print(int(shoulderLimitVar.get()))

        deviation, title, _ = smoothSeparate(comparing=comparingVar.get(),compareWithNoised=compareWithNoisedVar.get(), x=x, y=y, y_real=y_real,
                                             yx=yx, yx2=yx2, x2=x2, x3=x3, x4=x4, func_lambd=func_lambd,
                                             table=table, plot1=plot1,
                                             shoulderVector=findBestShoulderVector(
                                                 compareWithNoised=compareWithNoisedVar.get(), x=x, y=y,
                                                 y_real=y_real, yx=yx, yx2=yx2, x2=x2, x3=x3, x4=x4,
                                                 func_lambd=func_lambd, table=table,
                                                 generateWhole=generateWholeVectorVar.get(),
                                                 accuracy=float(accuracyVar.get()),
                                                 mode=optimizationMethodVar.get(),
                                                 path='../dots.xlsx',
                                                 smoothEdges=smoothVar.get(),
                                                 deviationType=deviationTypeVar.get(),
                                                 countEdges=сountEdgesVar.get(),
                                                 realtime=realTimeVar.get(),
                                                 progress=progress, window=window,
                                                 savepath=None,
                                                 iterations=int(iterationsVar.get()),
                                                 populationSize=int(populationSizeVar.get()),
                                                 mutationProb=float(mutationProbVar.get()),
                                                 shoulderLimit=int(shoulderLimitVar.get())),
                                             smoothEdges=smoothVar.get(), showRealFunction=showRealFuncVar.get(),
                                             countEdges=сountEdgesVar.get(), deviationType=deviationTypeVar.get(),
                                             realtime=realTimeVar.get(), func=func,
                                             amortizeSides=amortizeSidesVar.get(), sideLength=sideLengthVar.get(),linePlot=linePlotVar.get())
    # errorLabel.config(text="Отклонение: " + str(deviation))
    # plt.show(block=False)

    if savePlotsVar.get():
        if modeVar.get() == "1":
            mode = "fixedStep"
        elif modeVar.get() == "2":
            mode = "randomStep"
        elif modeVar.get() == "3":
            mode = "fixedStepAlternation"
        fullSavepath = savepath+"/"+functionVar.get().replace("\n", "").replace("/", "⁄").replace("**", "^").replace("*",
                                                                                                    "") + "_shoulder=" + str(
            shoulderVar.get()) + "_" + mode + "_" + str(intervalMinVar.get()) + "_to_" + str(
            intervalMaxVar.get()) + "_" + str(dotsCountVar.get()) + "_dots__noise_" + str(
            noiseMinVar.get()) if noiseVar.get() else "0" + "_to_" + str(
            noiseMaxVar.get()) if noiseVar.get() else "0"
        savepath = functionVar.get().replace("\n", "").replace("/", "⁄").replace("**", "^").replace("*",
                                                                                                    "") + "_shoulder=" + str(
            shoulderVar.get()) + "_" + mode + "_" + str(intervalMinVar.get()) + "_to_" + str(
            intervalMaxVar.get()) + "_" + str(dotsCountVar.get()) + "_dots__noise_" + str(
            noiseMinVar.get()) if noiseVar.get() else "0" + "_to_" + str(
            noiseMaxVar.get()) if noiseVar.get() else "0"

        fig1.savefig(fullSavepath + ".png")

        src_path = "../settings.ini"
        dst_path = "../inis/" + savepath + ".ini"
        shutil.copy(src_path, dst_path)
        src_path = "../dots.xlsx"
        dst_path = "../dots/" + savepath + ".xlsx"
        shutil.copy(src_path, dst_path)
        # print('Copied')

    config = configparser.ConfigParser()
    config.read("../settings.ini")
    if config["Generator"]["mode"] == "1":
        stepfortitle = "Равномерный, " + str("{:.4f}".format(
            (float(config["Generator"]["intervalMax"]) - float(config["Generator"]["intervalMin"])) / int(
                config["Generator"]["dotsCount"])))
    elif config["Generator"]["mode"] == "2":
        stepfortitle = "Случайный от " + str(config["Generator"]["randomStepMin"]) + " до " + str(
            config["Generator"]["randomStepMax"])
    elif config["Generator"]["mode"] == "3":
        stepfortitle = "Промежутки: " + str(
            config["Generator"]["segmentsCount"]) + " с шансом полного заполнения " + str(config["Generator"][
                                                                                              "percentageToFillSegment"]) + "%\n" + f"Частичное заполнение: "  # +str(config["Generator"]["percentToFillEmpty"]) +"%"

    noisefortitle = "Нет"
    if config["Generator"]["noise"] == "1":
        if config["Generator"]["distribution"] == "uniform":
            if config["Generator"]["noiseRelative"] == "1":
                noisefortitle = "Равномерный от " + config["Generator"]["minNoise"] + "% до " + config["Generator"][
                    "maxNoise"] + "%"
            else:
                noisefortitle = "Равномерный от " + config["Generator"]["minNoise"] + " до " + config["Generator"][
                    "maxNoise"]
        elif config["Generator"]["distribution"] == "normal":
            noisefortitle = "Нормальный, центр распределения: " + config["Generator"][
                "normalDistMean"] + ", дисперсия: " + config["Generator"]["normalDistDispersion"]
        if config["Generator"]["superNoises"] == "1":
            if config["Generator"]["superNoiseRelative"] == "1":
                noisefortitle += "\nВыбросы от " + config["Generator"]["superNoiseMin"] + "% до " + config["Generator"][
                    "superNoiseMax"] + "%"
            else:
                noisefortitle += "\nВыбросы от " + config["Generator"]["superNoiseMin"] + " до " + config["Generator"][
                    "superNoiseMax"]
            noisefortitle += ", процент выбросов: " + config["Generator"]["superNoisesPercent"] + "%"

    if optimizationMethodVar.get() == "Loop":
        optimizationmethodfortitle = "Полный перебор"
    elif optimizationMethodVar.get() == "Dichotomy":
        optimizationmethodfortitle = "Дихотомия"
    elif optimizationMethodVar.get() == "Random":
        optimizationmethodfortitle = "Случайный перебор, максимум итераций: " + str(
            iterationsVar.get()) + ", \nминимальная точность: " + str(accuracyVar.get())
    elif optimizationMethodVar.get() == "Genetic":
        optimizationmethodfortitle = "Генетический алгоритм, максимум популяций: " + str(
            iterationsVar.get()) + ",\nминимальная точность: " + str(
            accuracyVar.get()) + ", особей в популяции: " + str(
            populationSizeVar.get()) + ", вероятность мутации: " + str(float(mutationProbVar.get()) * 100) + "%"
    else:
        optimizationmethodfortitle = None

    if сountEdgesVar.get() == 1:
        if sideShoulderTypeVar.get() == "Increment":
            sidesForTitle = "Инкремент"
        elif sideShoulderTypeVar.get() == "Amortization":
            sidesForTitle = "Амортизация"

    else:
        sidesForTitle = "Не учитывается"
    title = config["Generator"]["func"] + ",\n" + config["Generator"][
        "dotsCount"] + " точек, шаг: " + stepfortitle + ", шум: " + noisefortitle + ", " + title + (
                f", Сглаживание краёв: {sidesForTitle}" if shoulderTypeVar.get() != "SeparateOptimal" else "") + (
                ", алгоритм: " + optimizationmethodfortitle if shoulderTypeVar.get() == "SeparateOptimal" else "") + ",\n затраченное время: " + str(
        "{:.2f}".format(time.time() - beginTime)) + " секунд(ы)"
    plot1.set_title(title, fontsize=10)
    canvas1.draw()
    canvas2.draw()


def on_button_click2():
    if modeVar.get() == "1":
        mode = "fixedStep"
    elif modeVar.get() == "2":
        mode = "randomStep"
    elif modeVar.get() == "3":
        mode = "fixedStepAlternation"

    # saveDirectoryVar.set(functionVar.get().replace("/", "⁄").replace("**", "^").replace("*", "") + "_shoulder=" + str(
    #     shoulderVar.get()) + "_" + mode + "_" + str(intervalMinVar.get()) + "_to_" + str(
    #     intervalMaxVar.get()) + "_" + str(dotsCountVar.get()) + "_dots__noise_" + (
    #                          str(noiseMinVar.get()) if noiseVar.get() else "0") + "_to_" + (
    #                          str(noiseMaxVar.get()) if noiseVar.get() else "0"))
    saveGeneratorSettings()
    Approximator.working = True

    if noiseVar.get() == 1:
        if superNoiseVar.get() == 1:
            generate(isRandomStep=randomStepVar.get(), minFillEmptyChance=minPercentToFillEmptyVar.get(),
                     maxFillEmptyChance=maxPercentToFillEmptyVar.get(), maxStep=maxStepVar.get(),
                     minStep=minStepVar.get(), mean=float(normalMeanVar.get()), stddev=float(normalDispersionVar.get()),
                     noiseType=distVar.get(), intervalCount=int(segmentCountVar.get()),
                     percentToFillSegment=float(percentToFillSegmentVar.get()), noisesRelative=noisesRelativeVar.get(),
                     superNoisesRelative=superNoisesRelativeVar.get(), plot1=plot1, table=table, func=functionVar.get(),
                     n=int(dotsCountVar.get()), start=float(intervalMinVar.get()), end=float(intervalMaxVar.get()),
                     mode=int(modeVar.get()), minNoise=float(noiseMinVar.get()), maxNoise=float(noiseMaxVar.get()),
                     superNoisesActivated=int(superNoiseVar.get()), superNoisesMin=float(superNoiseMinVar.get()),
                     superNoisesMax=float(superNoiseMaxVar.get()), superNoisePercent=int(superNoisePercentVar.get()))
        else:
            generate(isRandomStep=randomStepVar.get(), minFillEmptyChance=minPercentToFillEmptyVar.get(),
                     maxFillEmptyChance=maxPercentToFillEmptyVar.get(), maxStep=maxStepVar.get(),
                     minStep=minStepVar.get(), mean=float(normalMeanVar.get()), stddev=float(normalDispersionVar.get()),
                     noiseType=distVar.get(), intervalCount=int(segmentCountVar.get()),
                     percentToFillSegment=float(percentToFillSegmentVar.get()), noisesRelative=noisesRelativeVar.get(),
                     superNoisesRelative=superNoisesRelativeVar.get(), plot1=plot1, table=table, func=functionVar.get(),
                     n=int(dotsCountVar.get()), start=float(intervalMinVar.get()),
                     end=float(intervalMaxVar.get()), mode=int(modeVar.get()), minNoise=float(noiseMinVar.get()),
                     maxNoise=float(noiseMaxVar.get()))
    else:
        generate(isRandomStep=randomStepVar.get(), minFillEmptyChance=minPercentToFillEmptyVar.get(),
                 maxFillEmptyChance=maxPercentToFillEmptyVar.get(), maxStep=maxStepVar.get(), minStep=minStepVar.get(),noiseType=distVar.get(),
                 intervalCount=int(segmentCountVar.get()), percentToFillSegment=float(percentToFillSegmentVar.get()),
                 noisesRelative=noisesRelativeVar.get(), superNoisesRelative=superNoisesRelativeVar.get(), plot1=plot1,
                 table=table, func=functionVar.get(), n=int(dotsCountVar.get()), start=float(intervalMinVar.get()),
                 end=float(intervalMaxVar.get()), mode=int(modeVar.get()))
    canvas1.draw()


def onQueueButtonClick():
    with open("../file.txt", "r") as f:
        for line in f:
            functionVar.set(line)
            on_button_click2()
            on_button_click1()


def update_randomEnd_label(event):
    randomIntervalEndLabel.config(
        text=f"Примерный конец интервала: {dotsCountVar.get() * (minStepVar.get() + maxStepVar.get()) / 2 + intervalMinVar.get()}")


def updateUniformStepLabel(event):
    uniformStepLabel.config(
        text=f"Равномерный шаг: {(intervalMaxVar.get() - intervalMinVar.get()) / dotsCountVar.get()}")


# def on_button_click_changeSaveDirectory():
#   savepath = filedialog.asksaveasfilename(initialdir="inis")


# Create the main secondFrame
window = tk.Tk()
window.title("Модификация МНК")
# window.geometry("1920x1080")
# window.resizable(0,0)
window.state('zoomed')

notebookAndTable = tk.Frame(window)
notebook = ttk.Notebook(notebookAndTable)  # widget that manages a collection of windows/displays

tab1 = tk.Frame(notebook)  # new frame for tab 1
tab2 = tk.Frame(notebook)  # new frame for tab 2

notebook.add(tab1, text="Generator")
notebook.add(tab2, text="Approximator")

notebook.pack(anchor=tk.N)  # expand = expand to fill any space not otherwise used
notebook.select(tab2)
# fill = fill space on x and y axis


# mainFrame = tk.Frame(tab1)
# mainFrame.pack(fill=tk.BOTH,expand=1)
#
# # create the scrollCanvas and vertScrollbar inside the mainFrame using grid
# scrollCanvas = tk.Canvas(mainFrame)
# scrollCanvas.grid(row=0, column=0, sticky=tk.NSEW)
# vertScrollbar = ttk.Scrollbar(mainFrame, orient=tk.VERTICAL, command=scrollCanvas.yview)
# vertScrollbar.grid(row=0, column=1, sticky=tk.NS)
#
# # create the horScrollbar below the scrollCanvas using grid
# horScrollbar = ttk.Scrollbar(mainFrame, orient=tk.HORIZONTAL, command=scrollCanvas.xview)
# horScrollbar.grid(row=1, column=0, sticky=tk.EW)
#
# # configure the scrollCanvas to use the scrollbars
# scrollCanvas.configure(yscrollcommand=vertScrollbar.set,xscrollcommand=horScrollbar.set)
# scrollCanvas.bind('<Configure>', lambda e: scrollCanvas.configure(scrollregion=scrollCanvas.bbox("all")))
#
# secondFrame = tk.Frame(scrollCanvas)
# scrollCanvas.create_window((0,0), window=secondFrame, anchor="e")

# configure the rows and columns of the mainFrame to expand and fill the available space
# mainFrame.rowconfigure(0, weight=1)
# mainFrame.columnconfigure(0, weight=1)

approximatorFrame = tk.Frame(tab2, borderwidth=0.5, relief="solid")
approximatorLabel = tk.Label(approximatorFrame, text="Сглаживание", font=("Bold", 16), pady=10, borderwidth=2,
                             relief="solid")
approximatorLabel.pack(fill="x", anchor=tk.S)
firstColumnFrame = tk.Frame(approximatorFrame)
# Create a checkboxesFrame and add some widgets to it
plotSettingsLabel = tk.Label(firstColumnFrame, text="Параметры графиков", font=("Bold", 12))
plotSettingsLabel.pack()
plotSettingsFrame = tk.Frame(firstColumnFrame, borderwidth=0.5, relief="solid")
showRealFuncVar = tk.IntVar()
checkbox2 = tk.Checkbutton(plotSettingsFrame, text="Показать настоящую функцию", variable=showRealFuncVar)
checkbox2.pack(anchor=tk.W)
linePlotVar = tk.IntVar()
linePlotCheckbox = tk.Checkbutton(plotSettingsFrame, text="Соединить сглаженные точки", variable=linePlotVar)
linePlotCheckbox.pack(anchor=tk.W)
realTimeVar = tk.IntVar()
checkbox4 = tk.Checkbutton(plotSettingsFrame, text="Показать построения в реальном времени", variable=realTimeVar)
checkbox4.pack(anchor=tk.W)
savePlotsVar = tk.IntVar()
checkbox5 = tk.Checkbutton(plotSettingsFrame, text="Сохранять графики", variable=savePlotsVar)
checkbox5.pack(anchor=tk.W, side="left")
saveDirectoryVar = tk.StringVar()
currentSaveDirectoryEntry = tk.Entry(plotSettingsFrame, textvariable=saveDirectoryVar, borderwidth=0.5, relief="solid",
                                     width=35)
currentSaveDirectoryEntry.pack(side="left")
selectSaveDirectoryButton = tk.Button(plotSettingsFrame, text="Выбрать папку", command=lambda: saveDirectoryVar.set(
    filedialog.askdirectory(initialdir="")), relief="ridge", bd=3, height=1, width=12)
selectSaveDirectoryButton.pack(side="left", pady=2, padx=2)

plotSettingsFrame.pack(fill="x", padx=5)

checkboxesFrame = tk.Frame(firstColumnFrame, borderwidth=0.5, relief="solid")
smoothVar = tk.IntVar()


checkboxesLabel = tk.Label(firstColumnFrame, text="Параметры сглаживания", font=("Bold", 12))
checkboxesLabel.pack()

shoulderLimitFrame = tk.Frame(checkboxesFrame)
isShoulderLimitVar = tk.IntVar()
checkboxShoulderLimit = tk.Checkbutton(shoulderLimitFrame, text="Максимальная длина плеча", variable=isShoulderLimitVar,
                                       command=lambda: [shoulderLimitVar.set(999999), shoulderLimitEntry.config(
                                           state="normal" if isShoulderLimitVar.get() else "disabled")])
checkboxShoulderLimit.pack(anchor=tk.W, side="left")
shoulderLimitVar = tk.IntVar()
shoulderLimitEntry = tk.Entry(shoulderLimitFrame, textvariable=shoulderLimitVar, validate="key",
                              validatecommand=(shoulderLimitVar._register(validateInt), '%P'), borderwidth=0.5,
                              relief="solid", width=8)
shoulderLimitEntry.pack(side="left", anchor=tk.W)
shoulderLimitFrame.pack(anchor=tk.W)
comparingVar = tk.IntVar()
comparingCheckbox = tk.Checkbutton(checkboxesFrame, text="Показать другие методы", variable=comparingVar)
comparingCheckbox.pack(anchor=tk.W)


sideShoulderFrame = tk.Frame(checkboxesFrame)
checkbox1 = tk.Checkbutton(sideShoulderFrame, text="Сглаживание на концах", variable=smoothVar)
checkbox1.pack(anchor=tk.W, side="left")
sideShoulderTypeVar = tk.StringVar()
radioIncrementigShoulder = tk.Radiobutton(sideShoulderFrame, text="Инкремент", variable=sideShoulderTypeVar,
                                          value="Increment")
radioIncrementigShoulder.pack(side="left")
radioMovingAmortization = tk.Radiobutton(sideShoulderFrame, text="Амортизация", variable=sideShoulderTypeVar,
                                         value="Amortization")
radioMovingAmortization.pack(side="left")
sideShoulderFrame.pack(anchor=tk.W)


checkboxesFrame.pack(fill="x", padx=5, pady=5)

deviationLabel = tk.Label(firstColumnFrame, text="Отклонение", font=("Bold", 12))
deviationLabel.pack()
deviationFrame = tk.Frame(firstColumnFrame, borderwidth=0.5, relief="solid")
сountEdgesVar = tk.IntVar()
checkbox3 = tk.Checkbutton(deviationFrame, text="Учитывать концы при подсчёте погрешности", variable=сountEdgesVar)
checkbox3.pack(anchor=tk.W)
compareWithNoisedVar = tk.IntVar()
checkboxCompareWithNoised = tk.Checkbutton(deviationFrame, text="Сравнивать с заушмлёнными точками",
                                           variable=compareWithNoisedVar)
checkboxCompareWithNoised.pack(anchor=tk.W)
deviationTypeVar = tk.StringVar()
radioSquares = tk.Radiobutton(deviationFrame, text="Сумма квадратов", variable=deviationTypeVar, value="Squares")
radioSquares.pack(anchor=tk.W)
radioAbsolute = tk.Radiobutton(deviationFrame, text="Сумма модулей", variable=deviationTypeVar, value="Absolute")
radioAbsolute.pack(anchor=tk.W)
radioAverageSquares = tk.Radiobutton(deviationFrame, text="Среднеквадратичное отклонение", variable=deviationTypeVar,
                                     value="AverageSquares")
radioAverageSquares.pack(anchor=tk.W)

deviationFrame.pack(fill="x", padx=5,pady=12)

secondColumnFrame = tk.Frame(approximatorFrame)
shoulderTypeLabel = tk.Label(secondColumnFrame, text="Тип плеча", font=("Bold", 12))
shoulderTypeLabel.pack()
shoulderFrame = tk.Frame(secondColumnFrame, borderwidth=0.5, relief="solid", padx=5, pady=5)
shoulderTypeVar = tk.StringVar()
fixedShoulderFrame = tk.Frame(shoulderFrame)
radioFixedShoulder = tk.Radiobutton(fixedShoulderFrame, text="Заданное фикисрованное плечо", variable=shoulderTypeVar,
                                    value="Fixed", command=lambda: [shoulderEntry.configure(state="normal"),
                                                                    toggleFrame(optimizationFrame, 0),
                                                                    toggleFrame(fixedOptimalFrame, 0),
                                                                    optimizationParametersLabel.configure(
                                                                        state="disable")])
radioFixedShoulder.pack(side="left")
shoulderLenFrame = tk.Frame(fixedShoulderFrame)
shoulderVar = tk.IntVar()
shoulderEntry = tk.Entry(shoulderLenFrame, textvariable=shoulderVar, validate="key",
                         validatecommand=(shoulderVar._register(validateInt), '%P'), borderwidth=0.5, relief="solid",
                         width=6)
shoulderEntry.pack(side="left", pady=5)
fixedShoulderFrame.pack(anchor=tk.W)
shoulderLenFrame.pack(anchor=tk.W)

radioFixedOptimal = tk.Radiobutton(shoulderFrame, text="Оптимальное фиксированное плечо", variable=shoulderTypeVar,
                                   value="FixedOptimal", command=lambda: [shoulderEntry.configure(state="disable"),
                                                                          toggleFrame(fixedOptimalFrame, 1),
                                                                          toggleFrame(dichotomyFrame,
                                                                                      0 if fixedOptimalAlgVar.get() == "Loop" else 1),
                                                                          toggleFrame(optimizationFrame, 0),
                                                                          optimizationParametersLabel.configure(
                                                                              state="disable")])
radioFixedOptimal.pack(anchor=tk.W)
fixedOptimalFrame = tk.Frame(shoulderFrame, borderwidth=0.5, relief="solid")
fixedOptimalAlgVar = tk.StringVar()
radioFixedAlgLoop = tk.Radiobutton(fixedOptimalFrame, text="Последовательный перебор", variable=fixedOptimalAlgVar,
                                   value="Loop", command=lambda: toggleFrame(dichotomyFrame, 0))
radioFixedAlgLoop.pack(anchor=tk.W)

dichoFullFrame = tk.Frame(fixedOptimalFrame)
radioFixedAlgDichotomy = tk.Radiobutton(dichoFullFrame, text="Дихотомия", variable=fixedOptimalAlgVar,
                                        value="Dichotomy", command=lambda: toggleFrame(dichotomyFrame, 1))
radioFixedAlgDichotomy.pack(anchor=tk.W, side="left")
dichotomyFrame = tk.Frame(dichoFullFrame)
minDichotomyVar = tk.IntVar()
minDichotomyLabel = tk.Label(dichotomyFrame, text="от")
minDichotomyLabel.pack(side="left", pady=5)
minDichotomyEntry = tk.Entry(dichotomyFrame, textvariable=minDichotomyVar, validate="key",
                             validatecommand=(shoulderVar._register(validateInt), '%P'), borderwidth=0.5,
                             relief="solid", width=6)
minDichotomyEntry.pack(side="left", pady=5)
maxDichotomyLabel = tk.Label(dichotomyFrame, text="до")
maxDichotomyLabel.pack(side="left", pady=5)
maxDichotomyVar = tk.IntVar()
maxDichotomyEntry = tk.Entry(dichotomyFrame, textvariable=maxDichotomyVar, validate="key",
                             validatecommand=(shoulderVar._register(validateInt), '%P'), borderwidth=0.5,
                             relief="solid", width=6)
maxDichotomyEntry.pack(side="left", pady=5)
dichotomyFrame.pack(anchor=tk.W)
dichoFullFrame.pack(anchor=tk.W)

radioFixedAlgGolden = tk.Radiobutton(fixedOptimalFrame, text="Золотое сечение", variable=fixedOptimalAlgVar,
                                     value="Golden", command=lambda: toggleFrame(dichotomyFrame, 1))
radioFixedAlgGolden.pack(anchor=tk.W, side="left")
goldenFrame = tk.Frame(fixedOptimalFrame)
minGoldenVar = tk.IntVar()
minGoldenLabel = tk.Label(goldenFrame, text="от")
minGoldenLabel.pack(side="left", pady=5)
minGoldenEntry = tk.Entry(goldenFrame, textvariable=minGoldenVar, validate="key",
                          validatecommand=(shoulderVar._register(validateInt), '%P'), borderwidth=0.5, relief="solid",
                          width=6)
minGoldenEntry.pack(side="left", pady=5)
maxGoldenLabel = tk.Label(goldenFrame, text="до")
maxGoldenLabel.pack(side="left", pady=5)
maxGoldenVar = tk.IntVar()
maxGoldenEntry = tk.Entry(goldenFrame, textvariable=maxGoldenVar, validate="key",
                          validatecommand=(shoulderVar._register(validateInt), '%P'), borderwidth=0.5, relief="solid",
                          width=6)
maxGoldenEntry.pack(side="left", pady=5)
tauGoldenLabel = tk.Label(goldenFrame, text="t")
tauGoldenLabel.pack(side="left", pady=5)
tauGoldenVar = tk.DoubleVar()
tauGoldenEntry = tk.Entry(goldenFrame, textvariable=tauGoldenVar, validate="key",
                          validatecommand=(shoulderVar._register(validateInt), '%P'), borderwidth=0.5, relief="solid",
                          width=6)
tauGoldenEntry.pack(side="left", pady=5)
goldenFrame.pack(anchor=tk.W)

fixedOptimalFrame.pack(fill="x", padx=5)
separateRadioFrame = tk.Frame(shoulderFrame)
radioSeparateOptimal = tk.Radiobutton(separateRadioFrame, text="Оптимальное плечо для каждой точки",
                                      variable=shoulderTypeVar, value="SeparateOptimal",
                                      command=lambda: [shoulderEntry.configure(state="disable"),
                                                       toggleFrame(optimizationFrame, 1),
                                                       toggleFrame(fixedOptimalFrame, 0),
                                                       optimizationParametersLabel.configure(state="normal"),
                                                       toggleFrame(geneticFrame,
                                                                   1 if optimizationMethodVar.get() == "Genetic" else 0)])
radioSeparateOptimal.pack(anchor=tk.W)
amortizeSidesVar = tk.IntVar()
amortizeSidesCheckbox = tk.Checkbutton(separateRadioFrame, text="Амортизация краёв", variable=amortizeSidesVar)
amortizeSidesCheckbox.pack(anchor=tk.W)
sideLengthVar = tk.IntVar()
sideLengthLabel = tk.Label(separateRadioFrame, text="Длина края")
sideLengthLabel.pack(side="left", pady=5)
sideLengthEntry = tk.Entry(separateRadioFrame, textvariable=sideLengthVar, validate="key",
                           validatecommand=(shoulderVar._register(validateInt), '%P'), borderwidth=0.5, relief="solid",
                           width=6)
sideLengthEntry.pack(side="left", pady=5)

separateRadioFrame.pack(anchor=tk.W)
noRealFunctionVar = tk.IntVar()
noRealFunctionCheckbox = tk.Checkbutton(shoulderFrame, text="Не использовать настоящую функцию",
                                        variable=noRealFunctionVar)
noRealFunctionCheckbox.pack(anchor=tk.W)
optimizationParametersLabel = tk.Label(shoulderFrame, text="Параметры вычисления оптимального плеча", font=("Bold", 12))
optimizationParametersLabel.pack()
optimizationFrame = tk.Frame(shoulderFrame, borderwidth=0.5, relief="solid", pady=5)
generateWholeVectorVar = tk.IntVar()

generateWholeVectorCheckbox = tk.Checkbutton(optimizationFrame, text="Генерировать вектор целиком",
                                             variable=generateWholeVectorVar,
                                             command=lambda: [toggleFrame(dynamicDichotomyButtonFrame,
                                                                         0 if generateWholeVectorVar.get() else 1), toggleFrame(dynamicLoopButtonFrame,0 if generateWholeVectorVar.get() else 1)])
generateWholeVectorCheckbox.pack(anchor=tk.W)

iterationsFrame = tk.Frame(optimizationFrame)
iterationsLabel = tk.Label(iterationsFrame, text="Максимальное число итераций(популяций)", padx=3)
iterationsLabel.pack(side="left")
iterationsVar = tk.IntVar()
iterationsEntry = tk.Entry(iterationsFrame, textvariable=iterationsVar, validate="key",
                           validatecommand=(shoulderVar._register(validateInt), '%P'), borderwidth=0.5, relief="solid",
                           width=8)
iterationsEntry.pack()
iterationsFrame.pack(anchor=tk.W)

accuracyFrame = tk.Frame(optimizationFrame)
accuracyLabel = tk.Label(accuracyFrame, text="Точность", padx=3)
accuracyLabel.pack(side="left")
accuracyVar = tk.DoubleVar()
accuracyEntry = tk.Entry(accuracyFrame, textvariable=accuracyVar, validate="key",
                         validatecommand=(shoulderVar._register(validateFloat), '%P'), borderwidth=0.5, relief="solid",
                         width=8)
accuracyEntry.pack()
accuracyFrame.pack(anchor=tk.W, pady=5)

optimizationTypeLabel = tk.Label(optimizationFrame, text="Алгоритм поиска", font=("Bold", 12))
optimizationTypeLabel.pack()
optimizationTypeFrame = tk.Frame(optimizationFrame, borderwidth=0.5, relief="solid")
optimizationMethodVar = tk.StringVar()

dynamicLoopButtonFrame = tk.Frame(optimizationTypeFrame)
radioLoop = tk.Radiobutton(dynamicLoopButtonFrame, text="Перебор всех вариантов", variable=optimizationMethodVar,
                           value="Loop", command=lambda: [toggleFrame(geneticFrame, 0),
                                                          generateWholeVectorCheckbox.configure(state="normal")])
radioLoop.pack(anchor=tk.W)
dynamicLoopButtonFrame.pack(anchor=tk.W)
radioRandom = tk.Radiobutton(optimizationTypeFrame, text="Случайный перебор", variable=optimizationMethodVar,
                             value="Random", command=lambda: [toggleFrame(geneticFrame, 0),
                                                              generateWholeVectorCheckbox.configure(state="normal")])
radioRandom.pack(anchor=tk.W)
dynamicDichotomyButtonFrame = tk.Frame(optimizationTypeFrame)
radioDynamicDichotomy = tk.Radiobutton(dynamicDichotomyButtonFrame, text="Дихотомия", variable=optimizationMethodVar,
                                       value="Dichotomy", command=lambda: [toggleFrame(geneticFrame, 0),
                                                                           generateWholeVectorCheckbox.configure(
                                                                               state="disabled")])
radioDynamicDichotomy.pack(anchor=tk.W)
dynamicDichotomyButtonFrame.pack(anchor=tk.W)

radioGenetic = tk.Radiobutton(optimizationTypeFrame, text="Генетический алгоритм", variable=optimizationMethodVar,
                              value="Genetic", command=lambda: [toggleFrame(geneticFrame, 1),
                                                                generateWholeVectorCheckbox.configure(state="normal")])
radioGenetic.pack(anchor=tk.W, side="left")
geneticFrame = tk.Frame(optimizationTypeFrame)
populationFrame = tk.Frame(geneticFrame)
populationSizeVar = tk.IntVar()
populationSizeLabel = tk.Label(populationFrame, text="Особей в\n популяции")
populationSizeLabel.pack(side="left")
populationSizeEntry = tk.Entry(populationFrame, textvariable=populationSizeVar, validate="key",
                               validatecommand=(shoulderVar._register(validateInt), '%P'), borderwidth=0.5,
                               relief="solid", width=8)
populationSizeEntry.pack(side="left", padx=2)
populationFrame.pack(anchor=tk.W, side="left")

mutationProbVar = tk.DoubleVar()
mutationProbLabel = tk.Label(geneticFrame, text="Вероятность\n мутации")
mutationProbLabel.pack(anchor=tk.W, side="left")
mutationProbEntry = tk.Entry(geneticFrame, textvariable=mutationProbVar, validate="key",
                             validatecommand=(shoulderVar._register(validateFloat), '%P'), borderwidth=0.5,
                             relief="solid", width=8)
mutationProbEntry.pack(anchor=tk.W, side="left", padx=2)
geneticFrame.pack(anchor=tk.W, side="left")
optimizationTypeFrame.pack(fill="x", padx=5)

optimizationFrame.pack(fill="x")
shoulderFrame.pack(fill="x", padx=5)
approximateButton = tk.Button(firstColumnFrame, text="Построить график", font=("Bold", 14), command=on_button_click1,
                              relief="raised", bd=4)
approximateButton.pack(pady=2, expand=True, fill="x", padx=30)
stopButton = tk.Button(firstColumnFrame, text="Стоп", font=("Bold", 14), command=on_stop_button_click, relief="raised",
                       bd=4)
stopButton.pack(pady=2, expand=True, fill="x", padx=30)
# findBestShoulderButton = tk.Button(secondColumnFrame, text="Найти оптимальную длину плеча", command=on_button_click3, relief="raised", bd=4)
# findBestShoulderButton.pack(pady=5)
# bestShoulderLabel = tk.Label(secondColumnFrame, text="Оптимальное плечо:")
# bestShoulderLabel.pack()


progress = ttk.Progressbar(firstColumnFrame, orient="horizontal", length=200, mode="determinate")
progress.pack(pady=5, expand=True, fill="x", ipady=9, padx=30)

# set the progressbar to a certain value
progress.config(value=0)
secondColumnFrame.pack(side="left", anchor=tk.N)
firstColumnFrame.pack(anchor=tk.N, side="left")
approximatorFrame.pack(side="left", padx=5, pady=5, anchor=tk.N)

generatorFrame = tk.Frame(tab1, borderwidth=0.5, relief="solid")
generatorLabel = tk.Label(generatorFrame, text="Генерация точек", font=("Bold", 16), pady=10, borderwidth=2,
                          relief="solid")
generatorLabel.pack(fill="x", anchor=tk.N)

funcFrame = tk.Frame(generatorFrame, pady=2)
functionLabel = tk.Label(funcFrame, text="Функция х")
functionLabel.pack(side="left")
functionVar = tk.StringVar()
functionEntry = tk.Entry(funcFrame, textvariable=functionVar, borderwidth=0.5, relief="solid")
functionEntry.pack()
funcFrame.pack(anchor=tk.W, padx=5)

intervalFrame = tk.Frame(generatorFrame)
intervalLabel1 = tk.Label(intervalFrame, text="Интервал от")
intervalLabel1.pack(side="left")

intervalMinVar = tk.DoubleVar()
intervalMinEntry = tk.Entry(intervalFrame, textvariable=intervalMinVar, borderwidth=0.5, relief="solid", width=8,
                            validate="key", validatecommand=(intervalMinVar._register(validateFloat), '%P'))
intervalMinEntry.bind("<KeyRelease>", update_randomEnd_label, add="+")
intervalMinEntry.bind("<KeyRelease>", updateUniformStepLabel, add="+")
intervalMinEntry.pack(side="left")
intervalLabel2 = tk.Label(intervalFrame, text="до")
intervalLabel2.pack(side="left")
intervalMaxVar = tk.DoubleVar()
intervalMaxEntry = tk.Entry(intervalFrame, textvariable=intervalMaxVar, borderwidth=0.5, relief="solid", width=8,
                            validate="key", validatecommand=(intervalMaxVar._register(validateFloat), '%P'))
intervalMaxEntry.pack(side="left")
intervalMaxEntry.bind("<KeyRelease>", updateUniformStepLabel, add="+")
intervalFrame.pack(anchor=tk.W, padx=5, pady=2)

dotsCountFrame = tk.Frame(generatorFrame)
dotsCountLabel = tk.Label(dotsCountFrame, text="Количество точек")
dotsCountLabel.pack(side="left")
dotsCountVar = tk.IntVar()
dotsCountEntry = tk.Entry(dotsCountFrame, textvariable=dotsCountVar, borderwidth=0.5, relief="solid", width=8,
                          validate="key", validatecommand=(shoulderVar._register(validateInt), '%P'))
dotsCountEntry.bind("<KeyRelease>", updateUniformStepLabel, add="+")
dotsCountEntry.bind("<KeyRelease>", update_randomEnd_label, add="+")
dotsCountEntry.pack(side="left")
uniformStepLabel = tk.Label(dotsCountFrame, text="Равномерный шаг:")
uniformStepLabel.pack(side="left")
dotsCountFrame.pack(anchor=tk.W, padx=5, pady=2)

modeLabel = tk.Label(generatorFrame, text="Распределение точек", font=("Bold", 12))
modeLabel.pack(fill="x")
modeVar = tk.StringVar()
modeVar.set("")
modeFrame = tk.Frame(generatorFrame, borderwidth=0.5, relief="solid")

radioFixedStep = tk.Radiobutton(modeFrame, text="Стандарт", variable=modeVar, value="1",
                                command=lambda: [toggleFrame(randomStepFrame, 0), toggleFrame(segmentedFrame, 0)])

radioFixedStep.pack(anchor=tk.W)

anotherFrame = tk.Frame(modeFrame)
radioSegmented = tk.Radiobutton(anotherFrame, text="Промежутками", variable=modeVar, value="3",
                                command=lambda: [toggleFrame(randomStepFrame, 0), toggleFrame(segmentedFrame, 1)])
radioSegmented.pack(anchor=tk.W, side="left")
segmentedFrame = tk.Frame(anotherFrame)
segmentCountLabel = tk.Label(segmentedFrame, text="Количество\n промежутков")
segmentCountLabel.pack(side="left")
segmentCountVar = tk.IntVar()
segmentCountEntry = tk.Entry(segmentedFrame, textvariable=segmentCountVar, borderwidth=0.5, relief="solid", width=8,
                             validate="key", validatecommand=(intervalMaxVar._register(validateFloat), '%P'))
segmentCountEntry.pack(side="left")
percentToFillSegmentLabel = tk.Label(segmentedFrame, text="Шанс заполнить\n промежуток")
percentToFillSegmentLabel.pack(side="left")
percentToFillSegmentVar = tk.DoubleVar()
percentToFillSegmentEntry = tk.Entry(segmentedFrame, textvariable=percentToFillSegmentVar, borderwidth=0.5,
                                     relief="solid", width=8, validate="key",
                                     validatecommand=(intervalMaxVar._register(validateFloat), '%P'))
percentToFillSegmentEntry.pack(side="left", padx="2")
# percentToFillEmptyLabel = tk.Label(segmentedFrame, text="Шанс заполнить\n промежуток")
# percentToFillEmptyLabel.pack(side="left")
# percentToFillEmptyVar = tk.DoubleVar()
# percentToFillEmptyEntry = tk.Entry(segmentedFrame,textvariable=percentToFillEmptyVar,borderwidth=0.5, relief="solid",width=8,validate="key", validatecommand=(intervalMaxVar._register(validateFloat), '%P'))
# percentToFillEmptyEntry.pack(side="left", padx="2")
percentToFillEmptyLabel = tk.Label(segmentedFrame, text="Заполненность\n неполного промежутка")
percentToFillEmptyLabel.pack(side="left")
fromPercentToFillEmptyLabel = tk.Label(segmentedFrame, text="от")
fromPercentToFillEmptyLabel.pack(side="left")
minPercentToFillEmptyVar = tk.DoubleVar()
minPercentToFillEmptyEntry = tk.Entry(segmentedFrame, textvariable=minPercentToFillEmptyVar, borderwidth=0.5,
                                      relief="solid", width=8, validate="key",
                                      validatecommand=(intervalMaxVar._register(validateFloat), '%P'))
minPercentToFillEmptyEntry.pack(side="left", padx="2")
toPercentToFillEmptyLabel = tk.Label(segmentedFrame, text="до")
toPercentToFillEmptyLabel.pack(side="left")
maxPercentToFillEmptyVar = tk.DoubleVar()
maxPercentToFillEmptyEntry = tk.Entry(segmentedFrame, textvariable=maxPercentToFillEmptyVar, borderwidth=0.5,
                                      relief="solid", width=8, validate="key",
                                      validatecommand=(intervalMaxVar._register(validateFloat), '%P'))
maxPercentToFillEmptyEntry.pack(side="left", padx="2")
segmentedFrame.pack(anchor=tk.W)
anotherFrame.pack(anchor=tk.W)

frameBecauseOtherwiseItDisplaysWrong = tk.Frame(modeFrame)
randomStepVar = tk.IntVar()
checkRandom = tk.Checkbutton(frameBecauseOtherwiseItDisplaysWrong, text="Случайный шаг", variable=randomStepVar,
                             command=lambda: [toggleFrame(randomStepFrame, randomStepVar.get())])
checkRandom.pack(anchor=tk.W, side="left")
randomStepFrame = tk.Frame(frameBecauseOtherwiseItDisplaysWrong)

minStepVar = tk.DoubleVar()
maxStepVar = tk.DoubleVar()
minStepLabel = tk.Label(randomStepFrame, text="Шаг от")
minStepLabel.pack(side="left")
minStepEntry = tk.Entry(randomStepFrame, textvariable=minStepVar, borderwidth=0.5, relief="solid", width=8,
                        validate="key", validatecommand=(intervalMaxVar._register(validateFloat), '%P'))
minStepEntry.bind("<KeyRelease>", update_randomEnd_label, add="+")
minStepEntry.pack(side='left')
maxStepLabel = tk.Label(randomStepFrame, text="до")
maxStepLabel.pack(side="left")
maxStepEntry = tk.Entry(randomStepFrame, textvariable=maxStepVar, borderwidth=0.5, relief="solid", width=8,
                        validate="key", validatecommand=(intervalMaxVar._register(validateFloat), '%P'))
maxStepEntry.bind("<KeyRelease>", update_randomEnd_label, add="+")
maxStepEntry.pack(side='left')

randomIntervalEndLabel = tk.Label(randomStepFrame, text="Примерный конец интервала: ")
randomIntervalEndLabel.pack()

randomStepFrame.pack(anchor=tk.W, side="left")
frameBecauseOtherwiseItDisplaysWrong.pack(anchor=tk.W)

modeFrame.pack(fill="x", padx=5)

noiseVar = tk.IntVar()
checkboxNoise = tk.Checkbutton(generatorFrame, text="Шум", variable=noiseVar, font=("Bold", 12),
                               command=lambda: [toggleFrame(noisesFrame, noiseVar.get()),
                                                toggleFrame(superNoisesFrame, superNoiseVar.get() and noiseVar.get())])
checkboxNoise.pack()

noisesFrame = tk.Frame(generatorFrame, borderwidth=0.5, relief="solid")
noisesRelativeVar = tk.IntVar()
checkboxNoisesRelative = tk.Checkbutton(noisesFrame, text="Шум в процентах от изначального значения",
                                        variable=noisesRelativeVar)
checkboxNoisesRelative.pack()

distLabel = tk.Label(noisesFrame, text="Распределение шумов", font=("Bold", 11))
distLabel.pack()
distVar = tk.StringVar()
distVar.set("")
distFrame = tk.Frame(noisesFrame, borderwidth=0.5, relief="solid")

outerNormalFrame = tk.Frame(distFrame)
radioNormal = tk.Radiobutton(outerNormalFrame, text="Нормальное", variable=distVar, value="normal",
                             command=lambda: [toggleFrame(normalFrame, 1), toggleFrame(uniformDistFrame, 0)])
radioNormal.pack(anchor=tk.W, side="left")
normalFrame = tk.Frame(outerNormalFrame)
normalMeanVar = tk.DoubleVar()
normalMeanLabel = tk.Label(normalFrame, text="Центр\n распределения")
normalMeanLabel.pack(side="left")
normalMeanEntry = tk.Entry(normalFrame, textvariable=normalMeanVar, borderwidth=0.5, relief="solid", width=8,
                           validate="key", validatecommand=(intervalMinVar._register(validateFloat), '%P'))
normalMeanEntry.pack(side="left")
normalDispersionVar = tk.DoubleVar()
normalDispersionLabel = tk.Label(normalFrame, text="Дисперсия")
normalDispersionLabel.pack(side="left")
normalDispersionEntry = tk.Entry(normalFrame, textvariable=normalDispersionVar, borderwidth=0.5, relief="solid",
                                 width=8, validate="key",
                                 validatecommand=(intervalMinVar._register(validateFloat), '%P'))
normalDispersionEntry.pack(side="left")
normalFrame.pack(anchor=tk.W)
outerNormalFrame.pack(anchor=tk.W)

radioNeNormal = tk.Radiobutton(distFrame, text="Равномерное", variable=distVar, value="uniform",
                               command=lambda: [toggleFrame(normalFrame, 0), toggleFrame(uniformDistFrame, 1)])
radioNeNormal.pack(anchor=tk.W, side="left")
uniformDistFrame = tk.Frame(distFrame)
noiseValueFrame = tk.Frame(uniformDistFrame)
noiseLabel1 = tk.Label(noiseValueFrame, text="от")
noiseLabel1.pack(side="left")
noiseMinVar = tk.DoubleVar()
noiseMinEntry = tk.Entry(noiseValueFrame, textvariable=noiseMinVar, borderwidth=0.5, relief="solid", width=8,
                         validate="key", validatecommand=(intervalMinVar._register(validateFloat), '%P'))
noiseMinEntry.pack(side="left")
noiseLabel2 = tk.Label(noiseValueFrame, text="до")
noiseLabel2.pack(side="left")
noiseMaxVar = tk.DoubleVar()
noiseMaxEntry = tk.Entry(noiseValueFrame, textvariable=noiseMaxVar, borderwidth=0.5, relief="solid", width=8,
                         validate="key", validatecommand=(intervalMaxVar._register(validateFloat), '%P'))
noiseMaxEntry.pack(side="left")
noiseValueFrame.pack(pady=5)
uniformDistFrame.pack(anchor=tk.W, side="left")
distFrame.pack(fill="x", padx=5)

superNoiseVar = tk.IntVar()
checkboxSuperNoise = tk.Checkbutton(noisesFrame, text="Выбросы", variable=superNoiseVar,
                                    command=lambda: toggleFrame(superNoisesFrame, superNoiseVar.get()))
checkboxSuperNoise.pack()
superNoisesFrame = tk.Frame(noisesFrame, borderwidth=0.5, relief="solid")
superNoisePercentFrame = tk.Frame(superNoisesFrame)
superNoisePercentLabel = tk.Label(superNoisePercentFrame, text="Процент выбросов (%)")
superNoisePercentLabel.pack(side="left")
superNoisePercentVar = tk.DoubleVar()
superNoisePercentEntry = tk.Entry(superNoisePercentFrame, textvariable=superNoisePercentVar, borderwidth=0.5,
                                  relief="solid", width=3, validate="key",
                                  validatecommand=(intervalMinVar._register(validateFloat), '%P'))
superNoisePercentEntry.pack(side="left")
superNoisePercentFrame.pack(pady=3)
superNoisesRelativeVar = tk.IntVar()
checkboxSuperNoisesRelative = tk.Checkbutton(superNoisesFrame, text="Выбросы в процентах от\n изначального значения",
                                             variable=superNoisesRelativeVar)
checkboxSuperNoisesRelative.pack()
superNoisesValueFrame = tk.Frame(superNoisesFrame)
superNoiseLabel1 = tk.Label(superNoisesValueFrame, text="от")
superNoiseLabel1.pack(side="left")
superNoiseMinVar = tk.DoubleVar()
superNoiseMinEntry = tk.Entry(superNoisesValueFrame, textvariable=superNoiseMinVar, borderwidth=0.5, relief="solid",
                              width=8, validate="key", validatecommand=(intervalMinVar._register(validateFloat), '%P'))
superNoiseMinEntry.pack(side="left")
superNoiseLabel2 = tk.Label(superNoisesValueFrame, text="до")
superNoiseLabel2.pack(side="left")
superNoiseMaxVar = tk.DoubleVar()
superNoiseMaxEntry = tk.Entry(superNoisesValueFrame, textvariable=superNoiseMaxVar, borderwidth=0.5, relief="solid",
                              width=8, validate="key", validatecommand=(intervalMaxVar._register(validateFloat), '%P'))
superNoiseMaxEntry.pack(side="left")
superNoisesValueFrame.pack()

superNoisesFrame.pack(padx=5, pady=5, fill="x")
noisesFrame.pack(fill="x", padx=5)

generatorButtonsFrame = tk.Frame(generatorFrame)
generateButton = tk.Button(generatorButtonsFrame, text="Сгенерировать точки", font=("Bold", 14),
                           command=on_button_click2, relief="raised", bd=4)
generateButton.pack(pady=5, padx=5, side="left", fill="x", expand=True)
queueButton = tk.Button(generatorButtonsFrame, text="Запустить очередь", font=("Bold", 14), command=onQueueButtonClick,
                        bd=4)
queueButton.pack(side="left", fill="x", expand=True)
generatorButtonsFrame.pack(pady=10, padx=5, fill="x", expand=True)

figures = tk.Frame(window)
secondaryGraphFrame = tk.Frame(figures, borderwidth=0.5, relief="solid")
fig2 = mpl.figure.Figure(figsize=(11, 4), dpi=100)
fig2.tight_layout()
plot21 = fig2.add_subplot(121)
plot22 = fig2.add_subplot(122)

canvas2 = FigureCanvasTkAgg(fig2, master=secondaryGraphFrame)
canvas2.draw()
canvas2.get_tk_widget().pack()
# creating the Matplotlib toolbar
toolbar = NavigationToolbar2Tk(canvas2, secondaryGraphFrame)
toolbar.update()
# placing the toolbar on the Tkinter secondFrame
canvas2.get_tk_widget().pack()

# fig2 = mpl.figure.Figure(figsize=(1, 1), dpi=100)
# plot2 = #fig2.add_subplot(111)
# canvas2 = FigureCanvasTkAgg(#fig2,master=graphFrame)
# canvas2.draw()
# canvas2.get_tk_widget().pack()


generatorFrame.pack(side="top", padx=5, pady=5, anchor=tk.N, expand=True, fill="y")

# create the menu bar
menu_bar = tk.Menu(window)

# create the "File" menu
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Загрузить файл с параметрами",
                      command=lambda: loadSettings(filedialog.askopenfilename(initialdir="inis")))
file_menu.add_command(label="Сохранить параметры в файл",
                      command=lambda: saveAllSettings(filedialog.asksaveasfilename(initialdir="inis")))
file_menu.add_separator()
file_menu.add_command(label="Выход", command=window.quit)

# add the "File" menu to the menu bar
menu_bar.add_cascade(label="Меню", menu=file_menu)
window.config(menu=menu_bar)

tableFrame = tk.Frame(notebookAndTable, borderwidth=0.5, relief="solid")

table = ttk.Treeview(tableFrame, show="headings", columns=(1, 2, 3, 4, 5, 6, 7, 8), selectmode='browse')
style = ttk.Style(table)

table.heading('#1', text="X")
table.heading('#2', text="Y Real")
table.heading('#3', text="Y")
table.heading('#4', text="Y Smoothed")
table.heading('#5', text="YR - Y")
table.heading('#6', text="YR - YS")
table.heading('#7', text="Diff")
table.heading('#8', text="Shoulder")
table.column('#1', width=113, anchor=tk.CENTER)
table.column('#2', width=113, anchor=tk.CENTER)
table.column('#3', width=113, anchor=tk.CENTER)
table.column('#4', width=113, anchor=tk.CENTER)
table.column('#5', width=113, anchor=tk.CENTER)
table.column('#6', width=113, anchor=tk.CENTER)
table.column('#7', width=113, anchor=tk.CENTER)
table.column('#8', width=113, anchor=tk.CENTER)

# for i in range(3000):
#    table.insert("","end",values=[i,i+1,i+2,i+3, i+4, i+5,i+6, i+7])
style.configure('Treeview')

table.pack(side="left", fill="x", expand=True, anchor=tk.N)
vsb = ttk.Scrollbar(tableFrame, orient="vertical", command=table.yview)
vsb.pack(side="right", fill="y")
table.configure(yscrollcommand=vsb.set)
tableFrame.pack(anchor=tk.N, padx=5, pady=5)
table.config(height=17)

graphFrame = tk.Frame(figures, borderwidth=0.5, relief="solid")
fig1 = mpl.figure.Figure(figsize=(11, 6), dpi=100)  # (9.9, 9.42), dpi=100)
fig1.tight_layout()
plot1 = fig1.add_subplot(111)
fig1.subplots_adjust(top=0.845)

canvas1 = FigureCanvasTkAgg(fig1, master=graphFrame)
canvas1.draw()
canvas1.get_tk_widget().pack()
# creating the Matplotlib toolbar
toolbar = NavigationToolbar2Tk(canvas1, graphFrame)
toolbar.update()
# placing the toolbar on the Tkinter secondFrame
canvas1.get_tk_widget().pack()

# fig2 = mpl.figure.Figure(figsize=(1, 1), dpi=100)
# plot2 = #fig2.add_subplot(111)
# canvas2 = FigureCanvasTkAgg(#fig2,master=graphFrame)
# canvas2.draw()
# canvas2.get_tk_widget().pack()
graphFrame.pack(padx=5, pady=5, anchor=tk.W)
secondaryGraphFrame.pack(anchor=tk.W, padx=5)
notebookAndTable.pack(side="left", anchor=tk.N)
figures.pack(side="left", anchor=tk.N, pady=5)
##plot2.hist(x,y)
loadSettings()

# Run the main loop
window.mainloop()
