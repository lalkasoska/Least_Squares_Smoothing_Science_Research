n = int(input("power: "))
for j in range(n):
    print("a0" + "*sum(x" + str(j) + ") ", end="")
    for i in range(1,n):
        print("+ a"+str(i)+"*sum(x"+str(i+j)+") ", end ="")
    print("== sum(yx" + str(j)+")",end =", ")

for i in range(n):
    print("a"+str(i), end = " ")