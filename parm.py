# coding=utf-8
import os

PATH_PJ_ROOT = os.path.dirname(os.path.abspath(__file__))

PATH_DATA = os.path.join(PATH_PJ_ROOT, 'data')


PATH_DATA_TNEWS = os.path.join(PATH_DATA, 'tnews_public')
PATH_DATA_TNEWS_PRE = os.path.join(PATH_DATA_TNEWS, 'preprocessed')

PATH_MD_TMP = os.path.join(*[PATH_PJ_ROOT, 'wv_processor', 'modelfile'])

PATH_MD_FT = os.path.join(*[PATH_PJ_ROOT, 'classifier', 'vec_processor'])


# PATH_DATA_RAW = os.path.join(PATH_DATA, 'afqmc_public')
# PATH_DATA_PRE = os.path.join(PATH_DATA_RAW, 'preprocessed')
# COL_ST1 = 'sentence1'
# COL_ST2 = 'sentence2'
# COL_LB = 'label'
# COL_CLS = 'cls'
# COL_TXT = 'text'
# COL_CUT = 'cut'
