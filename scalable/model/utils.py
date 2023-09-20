from .abalone import Model as AbaloneModel
from .bankruptcy import Model as BankruptcyModel
from .cancer import Model as CancerModel
from .german_credit import Model as GermanModel
from .wine import Model as WineModel
from .stock_step0 import Model as StockModelStep0
from .stock_step1 import Model as StockModelStep1
from .stock import Model as StockModel
from .stock2 import Model as StockModel2
from .stock_ep import Model as StockModelEp
from .credit_card import Model as CreditModel
from .credit_card3 import Model as CreditModel3
from .credit_card_step2 import Model as CreditModel4


def get_model(data_name, model_name, tag=''):
    if data_name == 'abalone':
        return AbaloneModel(model_name)
    elif data_name == 'bankruptcy':
        return BankruptcyModel(model_name)
    elif data_name == 'stock_step0':
        return StockModelStep0(model_name)
    elif data_name == 'stock_step1':
        return StockModelStep1(model_name)
    elif data_name == 'stock':
        return StockModel(model_name)
    elif data_name == 'stockep':
        return StockModelEp(model_name)
    elif data_name == 'stock2':
        return StockModel2(model_name)
    elif data_name == 'credit' or data_name == 'Credit Card':
        return CreditModel(model_name)
    elif data_name == 'credit3' or data_name == 'Credit Card3':
        return CreditModel3(model_name)
    elif data_name == 'credit4' or data_name == 'Credit Card4':
        return CreditModel4(model_name)
    elif data_name == 'cancer':
        return CancerModel(model_name)
    elif data_name == 'german' or data_name == 'German Credit':
        return GermanModel(model_name)
    else:#if data_name == 'wine':
        return WineModel(model_name)
