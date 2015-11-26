from sklearn import datasets

X, y = datasets.make_moons(2000, noise=0.20)

# Can't believe I'm doing it this way, but join doesn't work
# on numpy strings and I'm on a plane unable to lookup the
# right way to join a column to a matrix and output as CSV.
for x_i,y_i in zip(X,y):
    output = ''
    output += str(y_i)
    for j in range(0,len(x_i)):
        output += ','
        output += str(x_i[j])
    print output

    
