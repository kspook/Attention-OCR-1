import platform

"""
Default paramters for experiemnt
"""


class ExpConfig:
    
    GPU_ID = 0
    # phase 
    #PHASE = 'test'
    PHASE = 'train'	
    VISUALIZE = True

    # input and output
    #DATA_BASE_DIR = '/mnt/90kDICT32px'
    #DATA_BASE_DIR = './evaluation_data/demo/'	
    #DATA_BASE_DIR = './test-img/'		
    #DATA_BASE_DIR = '../Attention-OCR-bank/image-data/'		
    #DATA_BASE_DIR = './image-data/'			
    DATA_BASE_DIR = './image-f59S/'				
    #DATA_PATH = '/mnt/train_shuffled_words.txt' # path containing data file names and labels. Format: 
    #DATA_PATH = './evaluation_data/demo/test.txt' # path containing data file names and labels. Format: 	
    #DATA_PATH = './test-img/test.txt' # path containing data file names and labels. Format: 		
    #DATA_PATH = '../Attention-OCR-bank/image-data/labels-map.csv' # path containing data file names and labels. Format: 
    DATA_PATH = './image-f59S/labels-map.csv' # path containing data file names and labels. Format: 	
    #MODEL_DIR = 'train' # the directory for saving and loading model parameters (structure is not stored)
    MODEL_DIR = 'trainf59S' # the directory for saving and loading model parameters (structure is not stored)	
    LOG_PATH = 'log_f59S.txt'
    #LOG_PATH = 'log_t2.txt'	
    OUTPUT_DIR = 'results_f59S' # output directory
    #OUTPUT_DIR = 'results_t2' # output directory	
    STEPS_PER_CHECKPOINT = 500 # checkpointing (print perplexity, save model) per how many steps

    # Optimization
    #NUM_EPOCH = 10
    NUM_EPOCH = 1000
    #BATCH_SIZE = 16
    BATCH_SIZE = 64
    INITIAL_LEARNING_RATE = 1.0 # initial learning rate, note the we use AdaDelta, so the initial value doe not matter much

    # Network parameters
    CLIP_GRADIENTS = True # whether to perform gradient clipping
    MAX_GRADIENT_NORM = 5.0 # Clip gradients to this norm
    TARGET_EMBEDDING_SIZE = 10 # embedding dimension for each target
    ATTN_USE_LSTM = True # whether or not use LSTM attention decoder cell
    ATTN_NUM_HIDDEN=128 # number of hidden units in attention decoder cell
    ATTN_NUM_LAYERS = 2 # number of layers in attention decoder cell
                        # (Encoder number of hidden units will be ATTN_NUM_HIDDEN*ATTN_NUM_LAYERS)
    #LOAD_MODEL = True
    LOAD_MODEL = False	
    OLD_MODEL_VERSION = False
    TARGET_VOCAB_SIZE = 13500 + 256+26+3+26+10+3 # 0: PADDING, 1: GO, 2: EOS, >2: 0-9, a-z,  han : 54620 shin 49888
    #TARGET_VOCAB_SIZE = 26+10+3 # 0: PADDING, 1: GO, 2: EOS, >2: 0-9, a-z
