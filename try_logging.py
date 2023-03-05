import logging

def logfun():
    logging.info('This is logging from a function.')
    print('This is printing from a function.')

if __name__=="__main__":
    logging.info('This is logging in main.')
    print('This is printing in main.')
    logfun()