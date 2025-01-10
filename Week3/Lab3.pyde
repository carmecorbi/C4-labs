######################################################
# Constant values for the simulations
ArraySize = 128
ConstantD = 0.00000325
ConstantK = 0.05
WorldSize = 0.05
delta_t = 1.0 / 100.0
delta_x = WorldSize / ArraySize
T1 = 0.115
T2 = 0.0085
##
######################################################
######################################################
# Question PW1
# Arrays
#We define the differents states
StateEDirichlet = [0.0, [0.0] * ArraySize,[0.0] * ArraySize] # Euler-Dirichlet state vector
StateRK2Dirichlet = [0.0, [0.0] * ArraySize,[0.0] * ArraySize] # RK2-Dirichlet state vector
StateENeumann = [0.0, [0.0] * ArraySize,[0.0] * ArraySize] # Euler-Neumann state vector
StateRK2Neumann = [0.0, [0.0] * ArraySize,[0.0] * ArraySize] # RK2-Neumann state vector
# Indices
StateTime = 0
StateConcentration = 1
StateCellType = 2
######################################################
######################################################
# Visualization Parameters
# Amount of pixels per cell on the X axis
PixelsPerCell = 6
# Dimensions of the visualization window
WindowWidth = PixelsPerCell * ArraySize
WindowHeight = WindowWidth / 2
##
######################################################
######################################################
# Question PW2
## Given an array of u_k values, return uxx=d2u/dx2
def approximate_uxx(array):
   uxx = [0.0]*ArraySize
   for i in range(1, ArraySize-1):
       uxx[i] = (array[i+1]- 2*array[i] + array[i-1])/sq(delta_x)
   return uxx
       

##
##########################################################

def SetInitialState():
    ######################################################
    # Question PW3
    ##
    # Your solution here
    ##
     # initial time is 0 s
    StateEDirichlet[StateCurrentTime] = 0.0 
    StateRK2Dirichlet[StateCurrentTime] = 0.0 
    StateENeumann[StateCurrentTime] = 0.0 
    StateRK2Neumann[StateCurrentTime] = 0.0 

    ######################################################

def EnforceBoundaryConditions(array, index, strategy):
    ######################################################
    # Question PW4
    ##
    # Dirichlet boundary conditions
    if strategy == 'Dirichlet':
        # Gene expression for 20 iterations
        array[index][0] = DIRICHLET_BOUNDARY # Boundary conditions of the first element are fix in Dirichlet
        array[index][-1] = DIRICHLET_BOUNDARY # Boundary conditions of the last element are fix in Dirichlet


    # Neumann boundary conditions
    elif strategy == 'Neumann':
        # Gene expression for 20 iterations
        array[index][0] = array[index][1]  # since v=0 in the boundary, first element must be equal to the following one
        array[index][-1] = array[index][-2]  # since v=0 in the boundary, last element must be equal to its previous one
    else:
        raise ValueError("Recheck strategy name!")
    ##
    ######################################################


def setup():
    SetInitialState()
    EnforceBoundaryConditions(StateEDirichlet, StateConcentration, 'Dirichlet')
    EnforceBoundaryConditions(StateRK2Dirichlet, StateConcentration, 'Dirichlet')
    EnforceBoundaryConditions(StateENeumann, StateConcentration, 'Neumann')
    EnforceBoundaryConditions(StateRK2Neumann, StateConcentration, 'Neumann')
    size(800, 860)
    
    WindowWidth = 800
    WindowHeight = 860

    colorMode(RGB, 1.0)
    noStroke()
    textSize(24)
    frameRate(1/delta_t)


def TimeStepE(array, strategy):
    ##################################################
    # Question PW5
    ##
    ## Update every state
    uxxs = approximate_uxx(us)
    ## Apply boundary conditions
    EnforceBoundaryConditions(array, StateConcentration, strategy)
    ## Update simulation time.
    array[StateCurrentTime] += delta_t 


def TimeStepRK2(array, strategy):
    ##################################################
    # Question PW5
    ##
    ## Update every state
    # Your code here

    ## Apply boundary conditions
    # Your code here
    
    ## Update simulation time.
    # Your code here
    ##
    ##################################################


def DrawState():
    OffsetX = 50
    OffsetY = 0.8 * WindowHeight
    TextBoxSize = 40

    for i in range(ArraySize):
        PixelsX = i * (PixelsPerCell - 1)
        
        # Question PW6   
        ##################################################
        ## Euler-Dirichlet Fill with the name of the index
        SimY = StateEDirichlet[___][i]
        fill(SimY, 0.0, (1 - SimY))
        rect(OffsetX + PixelsX, 220, PixelsPerCell - 1, -150 * SimY)
        #
        # Update cell types and fill() colors
        # Apply the rules on how cell types change!
        for a in range(1,5):
            circle(OffsetX+PixelsX+PixelsPerCell/2, 250-a*PixelsPerCell-1, PixelsPerCell-1)
        
        ## RK2-Dirichlet: Fill with the name of the index
        SimY = StateRK2Dirichlet[___][i]
        fill(SimY, 0.0, (1 - SimY))
        rect(OffsetX + PixelsX, 420, PixelsPerCell - 1, -150 * SimY)
        #
        # Update cell types and fill() colors
        # Apply the rules on how cell types change!
        for a in range(1,5):
            circle(OffsetX + PixelsX+PixelsPerCell/2, 450-a*PixelsPerCell-1, PixelsPerCell-1)
        
        ## Euler-Neumann: Fill with the name of the index
        SimY = StateENeumann[___][i]
        fill(SimY, 0.0, (1 - SimY))
        rect(OffsetX + PixelsX, 620, PixelsPerCell - 1, -150 * SimY)
        #
        # Update cell types and fill() colors
        # Apply the rules on how cell types change!   
        for a in range(1,5):
            circle(OffsetX + PixelsX+PixelsPerCell/2, 650-a*PixelsPerCell-1, PixelsPerCell-1)

        ## RK2-Neumann: Fill with the name of the index
        SimY = StateRK2Neumann[___][i]
        fill(SimY, 0.0, (1 - SimY))
        rect(OffsetX + PixelsX, 820, PixelsPerCell - 1, -150 * SimY)
        #
        # Update cell types and fill() colors
        # Apply the rules on how cell types change!
        for a in range(1,5):
            circle(OffsetX + PixelsX+PixelsPerCell/2, 850-a*PixelsPerCell-1, PixelsPerCell-1)

    # Protect the figure's name
    fill(1.0,1.0,1.0)
    rect(3.0,3.0,800-6,TextBoxSize)
    

def draw():
    background(0.9)

    TimeStepE(StateEDirichlet,'Dirichlet')
    TimeStepE(StateENeumann,'Neumann')
    TimeStepRK2(StateRK2Dirichlet,'Dirichlet')
    TimeStepRK2(StateRK2Neumann,'Neumann')
    DrawState()

    # Label.
    fill(0.0)
    text("French Flag model", 284, 32)
    text("Euler, Dirichlet", 300, 150)
    text("RK2, Dirichlet", 300, 350)
    text("Euler, Neumann", 300, 550)
    text("RK2, Neumann", 300, 750)
    
