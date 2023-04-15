import time
from ticker import Ticker
import data_op as op

'''start = time.time()
end = time.time()   
print(end - start)'''   

class SP500:


    def get_ticker(self, ticker_name):

        ticker_name = ticker_name.upper()

        if ticker_name in self.tickers_list:
            if self.sp500_dict[ticker_name] is None:
                self.sp500_dict[ticker_name] = Ticker(ticker_name)
            return self.sp500_dict[ticker_name]

        else:
            print('Ticker name not found')
            return


    def create_sp500_dict(self):

        empty_list = [None] * len(self.tickers_list)
        zipped = zip(self.tickers_list, empty_list)

        return dict(zipped)


    def __init__(self):

        self.tickers_list = op.get_sp500_tickers()
        self.sp500_dict = self.create_sp500_dict()



sp500 = SP500()

listt = sp500.tickers_list

'''start = time.time()
j = 0
for i in listt:
    starttt = time.time()
    sp500.get_ticker(i)
    enddd = time.time() 
    print(enddd - starttt, j)
    j +=1
end = time.time()   
print(end - start)'''