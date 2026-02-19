# Deep-Learning

Assignment 1

10-02-2026 
+ Pair of letters to classify is: F, N.

11-02-2026 
+ Seccond addition - Adding the base forward pass and updating of Bias and weights through stochastic gradient descent main issue solved, was missing setting j_running to 0 so model would never meet stopping conditions

17-02-2026 
+ third addition task 1 complete - issue fixed about datasets, was using more difficult dataset, expected not to converge

18-02-2026
+ ver 4 task 2 complete

+ ver 5 task 3 complete

















=========================== rough work =============================



#all of this is for batch gradient descent all the values are used
"""
#this will go into output layerself.layer1
        # basically the x for out output layer
        hiddenLayer = []

        # matrix multiplicaiton to avoid recalling the same funciton
        weightsHL = [[np.random.uniform(0,0.01) for _ in range(self.layer1)] for _ in range(X.shape[1])]

        #gives me two biases for our example one per nueron
        biasHL = [np.random.uniform(0,0.01) for _ in range(self.layer1)]

        # this returns an array 
        # cols = (eventually we will have a single col which will be the value that deteremines 0 or 1)
        # that is rows = inputs (we will need each entry to be able to recieve a class)
        z1 = np.dot(X, weightsHL) + biasHL
        a1 = logReg(z1)

        #z1 is the inputs from both nuerons into hidden layer
        # in this example 2 nuerons for the hidden layer 
        # each nuerons output is a feature and each row is a sample col being feature 1 feature 2
        weightsOL = [np.random.uniform(0,0.01) for _ in range(self.layer1)]

        biasOL=np.random.uniform(0,0.01)

        z2 = np.dot(a1, weightsOL) + biasOL
        a2 = logReg(z2)
"""

"""
        chosenIndex = np.random.randint(0,len(X))

        xi = X.iloc[chosenIndex]
        yi = Y.iloc[chosenIndex]

        z1 = np.dot(xi, weightsHL) + biasHL
        print("z1 :", z1 )
        a1 = logReg(z1)
        print("z1 :", z1 )
        print("a1 :", a1 )
        z2 = np.dot(a1, weightsOL)
        print("z2 :", z2 )
        a2 = logReg(z2)
        print("a2 :", a2 )
"""


"""
while(going):
            # CHoose random sample
            chosenIndex = np.random.randint(0,len(X))

            xi = X.iloc[chosenIndex]
            yi = Y.iloc[chosenIndex]

            # forward pass
            z1 = np.dot(xi, weightsHL) + biasHL
            a1 = logReg(z1)
            
            z2 = np.dot(a1, weightsOL)
            a2 = logReg(z2)

            J_current = lossfunction(yi, a2)

            #graident descent
            # initialise
            deltaWOL = [0] * X.shape[1]
            deltaBOL = 0

            deltaWHL = [[0 for _ in range(self.layer1)] for _ in range(X.shape[1])]
            
            #print(deltaWHL)
            # getting deltas ==================================
            dz2 = a2 - yi

            for j,_ in enumerate(weightsOL):
                deltaWOL[j] = dz2 * a1.iloc[j]

            deltaBOL = dz2

            for j,item in weightsHL:
                deltaWHL[j][0] = dz2 * weightsOL[j] * (a1[0] * (1 - a1[0])) 
                deltaWHL[j][1] = dz2 * weightsOL[j] * (a1[1] * (1 - a1[1])) 

            #for 

            # getting updated ====================================


            for j,_ in enumerate(weightsOL):
                weightsOL[j] = weightsOL[j] - (self.alpha * deltaWOL[j])

            biasOL = biasOL - (self.alpha * deltaBOL)







            


            # stoping critea =========
            iteration +=1
            if(iteration > self.max_iters):
                going = False

            J_running += J_current

            if (iteration%N == 0):
                if (abs(J_running - J_running_prev) < self.threshold):
                    going = False
                print(J_running, " for iteration ", iteration)
                J_running_prev = J_running
                J_running = 0;
"""


"""
# actual process =========================================
        
        chosenIndex = np.random.randint(0,len(X))

        xi = X.iloc[chosenIndex]
        xi = np.array(xi)
        yi = Y.iloc[chosenIndex]

        print("sample :" , xi)
        print("true answer :" , yi)

        # forward pass
        z1 = np.dot(xi, weightsHL) + biasHL
        a1 = logReg(z1)
        print("z1 :", z1 )
        print("a1 :", a1 )
            
        z2 = np.dot(a1, weightsOL)
        a2 = logReg(z2)
        print("z2 :", z2 )
        print("a2 :", a2 )
        print("")

        J_current = lossfunction(yi, a2)

        #graident descent

        #error for layer 2
        dz2 = a2 - yi
        #error for layer 1
        dz1 = dz2 * weightsOL * activaitonDer(z1)
        
        # initialise
        deltaWOL = [0] * X.shape[1]
        deltaWOL = np.array(deltaWOL)
        deltaBOL = 0

        print("delta weights ol shape:" , deltaWOL.shape)
        print("difference per nueron:" , dz1)

        
        #calculate deltas
        #output layer
        deltaWOL = a1 * dz2
        deltaBOL = dz2
        print(deltaWOL)

        #hidden layer
        deltaWHL = xi[:, None] * dz1[None, :] 
        deltaBHL = dz1

        print("delta hidden layers weights: " , deltaWHL)
        print("delta hidden layers bias:", deltaBHL)

        print("")
        print("")

        print("--------------------------------------")
        print("weights hl")
        print(weightsHL)
        print("bias hl")
        print(weightsHL)
        
        print("weights ol")
        print(weightsOL)
        print("bias ol")
        print(biasOL)
        print("--------------------------------------")

        

        #apply updates
        #output layer
        weightsOL = weightsOL - (self.alpha * deltaWOL)
        biasOL = biasOL - (self.alpha * deltaBOL)

        weightsHL = weightsHL - (self.alpha * deltaWHL)
        biasHL = biasHL - (self.alpha * deltaBHL)
        
        print("")
        print("")

        print("--------------------------------------")
        print("weights hl")
        print(weightsHL)
        print("bias hl")
        print(weightsHL)
        
        print("weights ol")
        print(weightsOL)
        print("bias ol")
        print(biasOL)
        print("--------------------------------------")
"""
