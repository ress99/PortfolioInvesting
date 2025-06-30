# pylint: skip-file

from asset_selection import AssetSelection
from portfolio_optimization import PortfolioOptimization
from index import DAX40

import mate as m
import mutate as mut
import evaluate as e
import select_ as s
import algorithm as a

import logging
import config_logger # pylint: disable=unused-import


logger = logging.getLogger(__name__)
logger.info("Starting Test")


GEN = 100
POP_SIZE = 50

def option_2():
    print('2')

def option_3():
    print('3')


dax = DAX40()

init_dict = {'generations': GEN, 'pop_size': POP_SIZE, 'indexes': [dax],
            'prtf_size': 10, 'start_date': '2012-01-01', 'end_date': '2012-12-31',
            'CXPB': 0.5, 'MUTPB': 0.5, 'objectives': (1, -1)}


def run_asset_selection():

    ea_dict = {'mate': m.as_mate,
            'mutate': mut.as_mutate_all, 
            'evaluate': e.return_risk,
            'select': s.selNSGA2, 
            'algorithm': a.base_algorithm}

    as_sel = AssetSelection(init_dict = init_dict)
    as_sel.ea_ops_from_dict(ea_dict)

    as_sel.init_population()
    as_sel.algorithm(as_sel)

    as_sel.final_prtf = as_sel.pareto_front[int(len(as_sel.pareto_front)/2)]

    print('\nAssets chosen:' + str(as_sel.final_prtf.asset_list))

    as_sel.plot_asset_returns()
    as_sel.plot_pareto_and_final_returns()
    as_sel.plot_objective_space()
    as_sel.plot_min_max()

    return as_sel


def test_asset_selection(as_sel):

    
    new_as_sel = as_sel.create_test_asset_selection(months = 6)

    new_as_sel.plot_asset_returns()
    new_as_sel.plot_pareto_and_final_returns(train = False)


def run_portfolio_optimization(as_sel):

    po = as_sel.create_portfolio_optimization()
    ea_dict = {'mate': m.po_mate,
            'mutate': mut.po_reverse_weights, 
            'evaluate': e.return_risk,
            'select': s.selNSGA2, 
            'algorithm': a.base_algorithm}

    po.ea_ops_from_dict(ea_dict)

    po.init_population()
    po.algorithm(po)

    po.final_prtf = po.pareto_front[int(len(po.pareto_front)/2)]

    print('\nWeights chosen:' + str(po.final_prtf.asset_weights))

    po.plot_asset_returns()
    po.plot_pareto_and_final_returns()
    po.plot_objective_space()
    po.plot_min_max()

    return po


def test_portfolio_optimization(po):

    new_as_sel = po.create_test_portfolio_optimization(months = 6)

    new_as_sel.plot_asset_returns()
    new_as_sel.plot_pareto_and_final_returns(train = False)



def main():

    as_sel = None
    po = None

    while True:

        print("\n=== Main Menu ===")
        print("1. Run Asset Selection")
        print("2. Test Asset Selection")
        print("3. Run Portfolio Optimization")
        print("4. Test Portfolio Optimization")
        # print("5. Update Data")
        print("e. Exit")

        choice = input("Select an option: ").strip().lower()
        print("\n")

        if choice == '1':
            as_sel = run_asset_selection()

        elif choice == '2':
            if as_sel is None:
                print("Please run Asset Selection first (option 1).")
            else:
                test_asset_selection(as_sel)

        elif choice == '3':
            if as_sel is None:
                print("Please run Asset Selection first (option 1).")
            else:
                po = run_portfolio_optimization(as_sel)

        elif choice == '4':
            if po is None:
                print("Please run Portfolio Optimization first (option 3).")
            else:
                test_portfolio_optimization(po)

        elif choice == 'e':
            print("Exiting program.")
            break
        else:
            print("Invalid option. Please try again.")

        

if __name__ == '__main__':
    main()