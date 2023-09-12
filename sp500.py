from ticker import Ticker
import data_op as op
 

#####
#Class used to 
class SP500:

    #####
    #Saves ticker inside the dictionary self.sp500_dict
    #####
    def get_ticker(self, ticker_name):

        ticker_name = ticker_name.upper()

        #If the ticker_name exists
        if ticker_name in self.tickers_list:
            #If the ticker is not yet in self.sp500_dict, initialize it
            if self.sp500_dict[ticker_name] is None:
                self.sp500_dict[ticker_name] = Ticker(ticker_name)
            return self.sp500_dict[ticker_name]

        #If the ticker_name is not valid
        else:
            print('Ticker name not found')
            return

    #####
    #Creates a dictionary where all the Ticker objects
    #Can be saved and accessed
    #####
    def create_sp500_dict(self):

        #Creates and empty list the size of the sp500
        empty_list = [None] * len(self.tickers_list)

        #Zip the tickers names and the empty list
        zipped = zip(self.tickers_list, empty_list)

        #Return as dictionary
        return dict(zipped)


    def __init__(self):

        self.tickers_list = op.get_sp500_tickers()
        self.sp500_dict = self.create_sp500_dict()


