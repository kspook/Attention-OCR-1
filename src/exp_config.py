import platform

"""
Default paramters for experiemnt
"""


class ExpConfig:
    
    GPU_ID = 0
    # phase 
    PHASE = 'test'
    #PHASE = 'train'	
    #PHASE = 'infer'		
    VISUALIZE = True

    # input and output
    #DATA_BASE_DIR = '/mnt/90kDICT32px'
    #DATA_BASE_DIR = './evaluation_data/demo/'	
    #DATA_BASE_DIR = 'word-img/'	
    #DATA_BASE_DIR = 'word-trainf8/'		
    #DATA_BASE_DIR = 'word-test/'		
    #DATA_BASE_DIR = '../Attention-OCR-bank/image-data/'
    #DATA_BASE_DIR = '../Attention-OCR-bank/image-f4S'	
    #DATA_BASE_DIR = './image-data/'	
    #DATA_BASE_DIR = 'image-data/image-f1S'		
    #DATA_BASE_DIR = './image-f59S/'
    #DATA_BASE_DIR = './image-f78S/'
    #DATA_BASE_DIR = './image-data/image-f78W/'	
    #DATA_BASE_DIR = './image-f64S/'
    #DATA_BASE_DIR = './image-data/image-f61S/'	
    #DATA_BASE_DIR = 'image-data/crpd_img/'		
    DATA_BASE_DIR = 'image-data/test-img3/'		
    #DATA_PATH = '/mnt/train_shuffled_words.txt' # path containing data file names and labels. Format: 
    #DATA_PATH = './evaluation_data/demo/test.txt' # path containing data file names and labels. Format: 	
    #DATA_PATH = 'image-data/test-img/test.txt' # path containing data file names and labels. Format: 		
    #DATA_PATH = '../Attention-OCR-bank/image-data/labels-map.csv' # path containing data file names and labels. Format: 
    #DATA_PATH = 'word-trainf8/labels-map.csv' # path containing data file names and labels. Format:
    #DATA_PATH = 'word-img/labels-map.csv' # path containing data file names and labels. Format:	
    #DATA_PATH = 'word-test/word-map.csv' # path containing data file names and labels. Format: 	
    #DATA_PATH = '../Attention-OCR-bank/image-f4S/labels-map.csv' # path containing data file names and labels. Format: 	
    #DATA_PATH = './image-f59S/labels-map.csv' # path containing data file names and labels. Format: 	
    #DATA_PATH = './image-f78S/labels-map.csv' # path containing data file names and labels. Format: 
    #DATA_PATH = './image-f78S/labels-map.csv' # path containing data file names and labels. Format: 	
    #DATA_PATH = './image-f64S/labels-map.csv' # path containing data file names and labels. Format: 
    #DATA_PATH = './image-data/image-f61S/labels-map.csv' # path containing data file names and labels. Format: 	
    #DATA_PATH = './image-data/image-f1S/labels-map.csv' # path containing data file names and labels. Format: 		
    #DATA_PATH = './image-data/testf61SE.txt' # path containing data file names and labels. Format: 	
    #DATA_PATH = './image-data/crpd_img/test.txt' # path containing data file names and labels. Format: 	
    #DATA_PATH = 'crpd_img/testw.txt' # path containing data file names and labels. Format: 
    DATA_PATH = 'image-data/test-img3/test.txt' # path containing data file names and labels. Format: 	
    #DATA_PATH = 'labels-map_int.csv' # path containing data file names and labels. Format: 	
    #MODEL_DIR = 'train' # the directory for saving and loading model parameters (structure is not stored)
    #MODEL_DIR = 'trainf59SD' # the directory for saving and loading model parameters (structure is not stored)
    #MODEL_DIR = 'trainf51W' # the directory for saving and loading model parameters (structure is not stored)
    #MODEL_DIR = 'trainf78S' # the directory for saving and loading model parameters (structure is not stored)
    #MODEL_DIR = 'trainf78W' # the directory for saving and loading model parameters (structure is not stored)	
    #MODEL_DIR = 'trainf8W' # the directory for saving and loading model parameters (structure is not stored)	
    #MODEL_DIR = 'trainf64SD' # the directory for saving and loading model parameters (structure is not stored)	
    #MODEL_DIR = 'trainf59SDR' # the directory for saving and loading model parameters (structure is not stored)		
    #MODEL_DIR = 'trainf59S' # the directory for saving and loading model parameters (structure is not stored)		
    #MODEL_DIR = 'trainf61SE' # the directory for saving and loading model parameters (structure is not stored)	
    MODEL_DIR = 'trainf61S' # the directory for saving and loading model parameters (structure is not stored)		
    #MODEL_DIR = 'trainf61Sint' # the directory for saving and loading model parameters (structure is not stored)		
    #MODEL_DIR = 'trainf1Sv2' # the directory for saving and loading model parameters (structure is not stored)		
    #LOG_PATH = 'log_f59SD.txt'
    #LOG_PATH = 'log_f78S.txt'
    #LOG_PATH = 'log_f78W.txt'	
    #LOG_PATH = 'log_f1Sv2.txt'
    #LOG_PATH = 'log_f61Sint.txt'	
    #LOG_PATH = 'log_f61SE.txt'	
    LOG_PATH = 'log_f61S.txt'		
    #LOG_PATH = 'log_f8W.txt'		
    #LOG_PATH = 'log_f59SDR.txt' 
    #LOG_PATH = 'log_f64SD.txt' 	
    #LOG_PATH = 'log_t2.txt'	
    #OUTPUT_DIR = 'results_f51W' # output directory
    #OUTPUT_DIR = 'results_f78S' # output directory
    #OUTPUT_DIR = 'results_f78W' # output directory
    #OUTPUT_DIR = 'results/results_f61Sint' # output directory	
    #OUTPUT_DIR = 'results_f1Sv2' # output directory	
    #OUTPUT_DIR = 'results_f1SE' # output directory		
    OUTPUT_DIR = 'results_f61SE' # output directory			
    #OUTPUT_DIR = 'results_f8W' # output directory	
    #OUTPUT_DIR = 'results_f59SD' # output directory	
    #OUTPUT_DIR = 'results_f64SD' # output directory	
    #OUTPUT_DIR = 'results_f59SDR' # output directory		
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
    LOAD_MODEL = True
    #LOAD_MODEL = False	
    OLD_MODEL_VERSION = False
    #TARGET_VOCAB_SIZE = 49
    #TARGET_VOCAB_SIZE = 26+4+10+3 # 0: PADDING, 1: GO, 2: EOS, >2: 0-9, a-z,  han : 54620 shin 49888	
    TARGET_VOCAB_SIZE = 6+26+6+12+3 # 0: PADDING, 1: GO, 2: EOS, >2: 0-9, a-z,  han : 54620 shin 49888
    #TARGET_VOCAB_SIZE = 26+10+3 # 0: PADDING, 1: GO, 2: EOS, >2: 0-9, a-z
    MAX_WIDTH=360
    MAX_HEIGHT=120	