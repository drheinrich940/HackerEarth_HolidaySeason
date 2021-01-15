IMAG_WIDTH = 80
IMG_HEIGHT = 300
IMG_DEPTH = 3
BATCH_SIZE = 32
NB_CLASSES = 6
H_FLIP = True
V_FLIP = False

#Improvements section. Add new improvement dedicated variable below, and add it to the dictionnary.
SE_MODULES = False
ASPP = False
SAM = False
IMPROVEMENTS = {'SE_MODULES': SE_MODULES, 'ASPP': ASPP, 'SAM': SAM}

#Loss and optimizer section
# Sparse Categorical Cross Entropy : SCCE
# Focal Loss : FL
# ArcFace : AF
LOSS = 'ADAM'
OPTI = 'SCCE'

EPOCHS = 10
SEED = 123

STAGES = (3, 6, 9)
FILTERS = (64, 128, 256, 512)
