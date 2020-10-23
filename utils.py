# coding=utf-8
import os
import logging
import logging.handlers

PATH_LOG =''

def init_logger(logger_name=__name__, level=logging.DEBUG):

    # if not os.path.exists(PATH_LOG):
    #     print(str(PATH_LOG) + 'creating!')
    #     os.makedirs(PATH_LOG)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # fmt
    basic_format ='%(asctime)s.%(msecs)03d [%(levelname)s] [%(name)s:%(lineno)s] - %(message)s'
    date_format = '%Y-%m-%d-%H:%M:%S'
    formatter = logging.Formatter(fmt=basic_format, datefmt=date_format)

    # hdr
    # file_for_save = os.path.join(PATH_LOG,'runtime.log')
    # file_handler = logging.handlers.RotatingFileHandler(filename=file_for_save, maxBytes=10*1024, backupCount=5,
    #                                                     encoding='utf-8')
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

if __name__ == '__main__':
    logger = init_logger(logger_name='qq')
    logger.info('ssss')




