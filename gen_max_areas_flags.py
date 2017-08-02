# Parameters of dataset
FILL_SHAPE = True       # Which set of primitives to use. filled or hollow.
DATA_NUM = 775            # Number of shapes to generate. From 1 to 775 (the max number of shapes in primitives set)
BATCH_SIZE = 2176 #87000

# Parameters of evolutionay algorithm
"""
To gain more precise results, you can increase POP_SIZE and NGEN. But it will cost more time.
When the POP_SIZE is small, I think it would be helpful to increase MUTPB appropriately,
in order to get rid of local optima.
"""
POP_SIZE = 1200          # The size of population. In this case, the number of sampling point on the shape.
NGEN = 20               # The number of generation, or iteration times.
CXPB = 0.5              # The probability of gene crossover.
MUTPB = 0.6             # The probability of gene mutation.
