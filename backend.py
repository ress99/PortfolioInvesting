# pylint: skip-file

from PyQt5 import QtCore, QtGui, QtWidgets
import os
import inspect
import importlib
import pickle


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import re
import matplotlib.pyplot as plt
# import numpy as np

import mate as m
import mutate as mut
import evaluate as e
import select_ as s
import algorithm as a

# pyuic5 -x test.ui -o tes.py
# from portfolio import Portfolio

from asset_selection import AssetSelection
from portfolio_optimization import PortfolioOptimization

# from main_tab import Ui_MainWindow
from tab_as import Ui_AS
from tab_po import Ui_PO

import config as c
import index
from data_op import get_path
# from deap import base
# from deap import creator
# from deap import tools



class Backend():

    #Setup Methods for Tabs
    def setup_create_object(self):

        self.create_index_dropdown()
        self.connect_object_buttons()
        self.setup_line_inputs()
        self.radio_buttons()


    def setup_run_algorithm(self):

        self.add_func_names()
        self.connect_algo_buttons()


    def setup_choose_individual(self):

        self.populate_combo_box_choose_plot()
        self.buttonPlot.clicked.connect(self.get_plot_choose_individual)
        self.checkBoxSimplified.stateChanged.connect(self.update_checkbox_prtfs)
        self.comboBoxChoosePlot.currentIndexChanged.connect(self.combo_choose_plot_changed)
        self.buttons_final_prtf()

    
    def setup_test_portfolios(self):

        self.populate_combo_box_choose_plot_test()
        self.buttonPlotTest.clicked.connect(self.get_plot_test_prtf)
        self.radio_test_buttons()
        self.comboBoxChoosePlotTest.currentIndexChanged.connect(self.combo_choose_plot_test_changed)
        self.connect_test_buttons()


    def choose_final_prtf(self):

        if self.status == 2:
            selected_ind = self.get_selected_prtf()
            if selected_ind:
                self.obj.final_prtf = selected_ind
                self.labelFinalPrtf.setText("Final Portfolio: " + str(self.obj.final_prtf.asset_list))
                self.update_status()
            else:
                self.show_popup('warning', 'Please select a valid Portfolio.')
        else:
            self.show_popup('warning', 'Cannot choose final Portfolio. Please remove the selected one first.')

    def remove_final_prtf(self):

        if self.status == 3:
            del self.obj.final_prtf
            self.labelFinalPrtf.setText("Please select a Portfolio")
            self.update_status()
        else:
            self.show_popup('warning', 'Cannot remove final Portfolio. Please Select one first.')


    def buttons_final_prtf(self):

        self.buttonChoosePortfolio.clicked.connect(self.choose_final_prtf)
        self.buttonRemovePortfolio.clicked.connect(self.remove_final_prtf)
        self.labelFinalPrtf.setWordWrap(True)


    def radio_buttons(self): 

        self.bb_mode = False
        self.evolutionary_algorithm_spins(False)
        
        self.radioEA = QtWidgets.QRadioButton("Evolutionary Algorithm")
        self.radioEA.setChecked(True)
        self.layoutEABB.addWidget(self.radioEA)
        self.radioEA.toggled.connect(lambda: self.radio_button_clicked(self.radioEA))

        self.radioBB = QtWidgets.QRadioButton("Black-Box")
        self.layoutEABB.addWidget(self.radioBB)
        self.radioBB.toggled.connect(lambda: self.radio_button_clicked(self.radioBB))


    def radio_button_clicked(self, button):
        if button.text() == "Evolutionary Algorithm":
            self.evolutionary_algorithm_spins(False)
        elif button.text() == "Black-Box":
            self.evolutionary_algorithm_spins(True)


    def evolutionary_algorithm_spins(self, bool):
        
        self.bb_mode = bool

        self.lineInputBBFile.setDisabled(not bool)

        self.doubleSpinCX.setDisabled(bool)
        self.doubleSpinMUT.setDisabled(bool)
        self.spinGenerations.setDisabled(bool)
        self.spinPopSize.setDisabled(bool)


    def radio_test_buttons(self): 

        self.radioTestPop = QtWidgets.QRadioButton("Population")
        self.radioTestPop.setChecked(True)
        self.layoutTestPRTFs.addWidget(self.radioTestPop)
        self.radioTestPop.toggled.connect(lambda: self.radio_button_clicked(self.radioTestPop))

        self.radioTestPF = QtWidgets.QRadioButton("Pareto Front")
        self.layoutTestPRTFs.addWidget(self.radioTestPF)
        self.radioTestPF.toggled.connect(lambda: self.radio_button_clicked(self.radioTestPF))

        self.radioTestFP = QtWidgets.QRadioButton("Final Portfolio")
        self.layoutTestPRTFs.addWidget(self.radioTestFP)
        self.radioTestFP.toggled.connect(lambda: self.radio_button_clicked(self.radioTestFP))


    def setup_line_inputs(self):

        regex = QtCore.QRegExp('[0-9,\s-]+')
        validator = QtGui.QRegExpValidator(regex)
        self.lineInputObjectives.setValidator(validator)
        self.lineInputObjectives.setPlaceholderText("CSV format")
        self.lineInputObjectives.setText("1, -1")
        
        regex = QtCore.QRegExp('[a-zA-Z/\\\\.]+')
        validator = QtGui.QRegExpValidator(regex)
        self.lineInputBBFile.setValidator(validator)
        self.lineInputBBFile.setPlaceholderText("e.g. blackbox.py")

        return


    def populate_combo_box_choose_plot(self):

        self.comboBoxChoosePlot.addItem("Multiple Objectives")
        self.comboBoxChoosePlot.addItem("All Portfolio Returns")
        self.comboBoxChoosePlot.addItem("Chosen Portfolio Assets")


    def combo_choose_plot_changed(self):

        text = self.comboBoxChoosePlot.currentText()
        if text == "Multiple Objectives":
            bool_ = False
        else:
            bool_ = True

        self.spinMinX.setDisabled(bool_)
        self.spinMaxX.setDisabled(bool_)
        self.spinMinY.setDisabled(bool_)
        self.spinMaxY.setDisabled(bool_)
        self.checkBoxBoundaries.setDisabled(bool_)


    def populate_combo_box_choose_plot_test(self):

        self.comboBoxChoosePlotTest.addItem("Portfolio Returns")
        self.comboBoxChoosePlotTest.addItem("Final Portfolio Assets")


    def combo_choose_plot_test_changed(self):

        text = self.comboBoxChoosePlotTest.currentText()
        if text == "Portfolio Returns":
            bool_ = False
        else:
            bool_ = True

        self.radioTestPop.setDisabled(bool_)
        self.radioTestPF.setDisabled(bool_)
        self.radioTestFP.setDisabled(bool_)


    def backend_init_pop(self):

        if self.status == 1:
            self.obj.init_population()
            self.show_popup('info', 'Population Initialized')


    def add_func_names(self):

        mate_list = self.get_methods('mate.py')
        [self.comboBoxMate.addItem(i) for i in mate_list]

        mutate_list = self.get_methods('mutate.py')
        [self.comboBoxMutate.addItem(i) for i in mutate_list]

        select_list = self.get_methods('select_.py')
        [self.comboBoxSelect.addItem(i) for i in select_list]

        evaluate_list = self.get_methods('evaluate.py')
        [self.comboBoxEvaluate.addItem(i) for i in evaluate_list]

        algo_list = self.get_methods('algorithm.py')
        [self.comboBoxAlgo.addItem(i) for i in algo_list]


    def backend_register_methods(self):

        if self.status == 1:
            self.obj.select = getattr(s, self.comboBoxSelect.currentText())

            self.obj.mate = getattr(m, self.comboBoxMate.currentText())
            
            self.obj.mutate = getattr(mut, self.comboBoxMutate.currentText())
            
            self.obj.evaluate = getattr(e, self.comboBoxEvaluate.currentText())

            self.obj.run_algorithm = getattr(a, self.comboBoxAlgo.currentText())

            self.show_popup('info', 'Evolutionary Operators Registered')


    def backend_run_algo(self):

        if self.status == 1:
            if hasattr(self.obj, 'select'):
                self.obj.run_algorithm(self.obj)
                self.show_popup('info', 'Algorithm Finished')
                self.update_status()


    def connect_object_buttons(self):
        self.buttonCreateObject.clicked.connect(self.create_object)
        self.buttonImportObject.clicked.connect(self.import_object)
        self.buttonDeleteObject.clicked.connect(self.delete_object)
        self.buttonSaveObject.clicked.connect(self.save_object)


    def connect_test_buttons(self):
        self.buttonCreateTest.clicked.connect(self.create_test_object)


    def clear_list_prtfs(self):
            
            self.comboBoxListAssets.clear()
            self.comboBoxListAssets.addItem("Select Portfolio")
    

    def update_checkbox_prtfs(self):

        self.clear_list_prtfs()
        if self.status >= 2:

            self.update_prtfs_on_display()

            for i in self.prtfs_on_display:
                if isinstance(self, Backend_AS):
                    self.comboBoxListAssets.addItem(str(list(i.asset_list)))
                else:
                    self.comboBoxListAssets.addItem(str(list(i.asset_weights)))


    def update_prtfs_on_display(self):

        if self.checkBoxSimplified.isChecked():
            self.prtfs_on_display = [self.obj.pareto_front[i] for i in [0, len(self.obj.pareto_front)//2, -1]]
        else:

            if isinstance(self, Backend_AS):
                aux_list = [i.asset_list for i in self.obj.pareto_front]
            else:
                aux_list = [i.asset_weights for i in self.obj.pareto_front]

            _, ids = self.remove_duplicates(aux_list)
            self.prtfs_on_display = [self.obj.pareto_front[i] for i in ids]


    def get_plot_choose_individual(self):
        
        if self.status >= 2:

            plt.close()
            self.clear_layout(self.layoutFig)
            self.canvasChooseIndividual = FigureCanvas(plt.figure())
            # self.toolbarChooseIndividual = NavigationToolbar(self.canvasChooseIndividual, self.verticalLayoutWidget_2)
            self.toolbarChooseIndividual = NavigationToolbar(self.canvasChooseIndividual, self.plotWidgetContainer)
            self.layoutFig.addWidget(self.toolbarChooseIndividual)
            self.layoutFig.addWidget(self.canvasChooseIndividual)
            
            ax = self.canvasChooseIndividual.figure.add_subplot(111)

            idx = self.comboBoxChoosePlot.currentIndex()
            if idx == 0:
                self.plot_MO(ax)
            elif idx == 1:
                self.plot_portfolio_returns_chosen_tab(ax)
            elif idx == 2:

                if hasattr(self.obj, 'final_prtf'):
                    prtf_to_plot = self.obj.final_prtf
                    title = 'Final Portfolio Assets'
                else:
                    prtf_to_plot = self.get_selected_prtf()
                    title = 'Selected Portfolio Assets'
                    if not prtf_to_plot:
                        prtf_to_plot = self.prtfs_on_display[0]
                        title = 'First Portfolio Assets'
                self.plot_asset_returns(ax, prtf_to_plot, title)

            self.canvasChooseIndividual.draw()


    def text_for_plot_choose_individual(self, text):

        self.clear_layout(self.layoutFig)

        self.labelPlot = QtWidgets.QLabel(text)
        font = QtGui.QFont()
        font.setPointSize(25)
        self.labelPlot.setFont(font)
        self.labelPlot.setAlignment(QtCore.Qt.AlignCenter)
        self.layoutFig.addWidget(self.labelPlot)


    def get_plot_test_prtf(self):

        if hasattr(self, 'test_obj'):
            plt.close()
            self.clear_layout(self.layoutFigTest)
            self.canvasTest = FigureCanvas(plt.figure())
            self.toolbarTest = NavigationToolbar(self.canvasTest, self.plotWidgetContestContainer)
            self.layoutFigTest.addWidget(self.toolbarTest)
            self.layoutFigTest.addWidget(self.canvasTest)
            
            ax = self.canvasTest.figure.add_subplot(111)

            idx = self.comboBoxChoosePlotTest.currentIndex()
            if idx == 0:
                if self.radioTestPop.isChecked():
                    self.plot_portfolio_returns_test_tab(ax, self.test_obj.pop)
                elif self.radioTestPF.isChecked():
                    self.plot_portfolio_returns_test_tab(ax, self.test_obj.pareto_front)
                elif self.radioTestFP.isChecked():
                    self.plot_portfolio_returns_test_tab(ax, [self.test_obj.final_prtf])
                # self.plot_portfolio_returns_test_tab(ax, self.test_obj.pareto_front)
            elif idx == 1:
                self.plot_asset_returns(ax, self.test_obj.final_prtf, 'Final Portfolio Assets over Test Period')

            self.canvasTest.draw()


    def text_for_plot_test(self, text):

        self.clear_layout(self.layoutFigTest)

        self.labelPlotTest = QtWidgets.QLabel(text)
        font = QtGui.QFont()
        font.setPointSize(25)
        self.labelPlotTest.setFont(font)
        self.labelPlotTest.setAlignment(QtCore.Qt.AlignCenter)
        self.layoutFigTest.addWidget(self.labelPlotTest)


    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
            else:
                sub_layout = item.layout()
                if sub_layout is not None:
                    self.clear_layout(sub_layout)


    def plot_portfolio_returns_chosen_tab(self, ax):

        selected_ind = self.get_selected_prtf()
        final_prtf = None
        if hasattr(self.obj, 'final_prtf'):
            final_prtf = self.obj.final_prtf
        if selected_ind or final_prtf:
            alpha = 0.65
        else:
            alpha = 1

        for ind in self.obj.pareto_front:

            df_to_plot = ind.get_portfolio_returns_df()
            ax.plot(df_to_plot.index, df_to_plot.values, color = "#A3A3A3", alpha = alpha)

        if selected_ind:
            df_to_plot = selected_ind.get_portfolio_returns_df()
            ax.plot(df_to_plot.index, df_to_plot.values, color = 'black', linewidth=2.0, label = 'Selected Portfolio')

        if final_prtf:
            df_to_plot = final_prtf.get_portfolio_returns_df()
            ax.plot(df_to_plot.index, df_to_plot.values, color = '#ff0000', linewidth=2.0, label = 'Final Portfolio')

        ax.set_xlabel('Time')
        ax.set_ylabel('Returns')
        ax.set_title('Portfolio Returns over Train')
        ax.grid(True)
        # ax.tick_params(axis='x', rotation=20)
        ax.figure.autofmt_xdate()
        if selected_ind or final_prtf:
            ax.legend()

        return   


    def plot_portfolio_returns_test_tab(self, ax, list_of_portfolios):

        for ind in list_of_portfolios:

            df_to_plot = ind.get_portfolio_returns_df()
            ax.plot(df_to_plot.index, df_to_plot.values, color = "#A3A3A3", alpha = 0.65)

        # for ind in self.test_obj.pareto_front:

        #     df_to_plot = ind.get_portfolio_returns_df()
        #     ax.plot(df_to_plot.index, df_to_plot.values, color = "#A3A3A3", alpha = 0.65)

        test_final_prtf = self.test_obj.final_prtf
        df_to_plot = test_final_prtf.get_portfolio_returns_df()
        ax.plot(df_to_plot.index, df_to_plot.values, color = '#ff0000', linewidth=2.0, label = 'Final Portfolio')

        index_list = test_final_prtf.index_objects
        index_prtf = test_final_prtf.get_index_portfolio(index_list)
        df_to_plot = index_prtf.get_portfolio_returns_df()
        ax.plot(df_to_plot.index, df_to_plot.values, color = "#0035e2", linewidth=2.0, label = 'Index Portfolio')

        ax.set_xlabel('Time')
        ax.set_ylabel('Returns')
        ax.set_title('Portfolio Returns over Test Period')
        ax.grid(True)
        # ax.tick_params(axis='x', rotation=-20)
        ax.figure.autofmt_xdate()
        ax.legend()

        return


    def plot_asset_returns(self, ax, individual, title):

        df = individual.prtf_df.copy()

        df.set_index('Date', inplace=True)
        df = df / df.iloc[0] * 100
        for col in df.columns:
            ax.plot(df.index, df[col], label=col)

        ax.set_xlabel('Time')
        ax.set_ylabel('Returns')
        ax.set_title(title)
        ax.grid(True)
        ax.figure.autofmt_xdate()
        ax.legend()

    # def remove_bounds(self):

    #     if not self.checkBoxBoundaries.isChecked():
    #         return self.obj.pareto_fronts
    #     else:
    #         min_values = [self.spinMinX.value(), self.spinMinY.value()]
    #         max_values = [self.spinMaxX.value(), self.spinMaxY.value()]
    #         arr_list = [None] * len(self.obj.pareto_fronts)
    #         for idx, arr in enumerate(self.obj.pareto_fronts):
    #             arr_list[idx] = np.array([i for i in arr if all(min_v <= val <= max_v for val, (min_v, max_v) in zip(i, zip(min_values, max_values)))])
    #         return arr_list

    def plot_MO(self, ax):

        min_values = [self.spinMinX.value(), self.spinMinY.value()]
        max_values = [self.spinMaxX.value(), self.spinMaxY.value()]
        arr_list = self.obj.remove_bounds(self.checkBoxBoundaries.isChecked(), min_values, max_values)

        for idx, arr in enumerate(arr_list):
            if len(arr) == 0:
                continue
            if idx == len(arr_list) - 1:
                ax.plot(arr[:, 0], arr[:, 1], color = 'blue', marker = 'o', label = 'Pareto Front')
            else:
                ax.plot(arr[:, 0], arr[:, 1], color = '#8dff52', marker = 'o', alpha = 0.8)


        selected_ind = self.get_selected_prtf()
        if selected_ind:
            prtf_point = selected_ind.fitness.values
            ax.plot(prtf_point[0], prtf_point[1], color = 'black', marker = 'o', label = 'Selected Portfolio')

        if hasattr(self.obj, 'final_prtf'):
            prtf_point = self.obj.final_prtf.fitness.values
            ax.plot(prtf_point[0], prtf_point[1], color = 'red', marker = 'o', label = 'Final Portfolio')


        ax.set_xlabel('X Axis Label')
        ax.set_ylabel('Y Axis Label')
        ax.set_title('Sine Wave Plot')
        ax.legend()
        ax.grid(True)

    def get_selected_prtf(self):
        combo_idx = self.comboBoxListAssets.currentIndex() - 1
        if combo_idx >= 0:         
            return self.prtfs_on_display[combo_idx]
        return False

    def get_selected_indexes(self):

        selected_indexes = [item.text() for item in self.listIndexes.selectedItems()]
        indexes = [self.index_dict[i] for i in selected_indexes]
        return indexes

    def check_input(self):
        input_text = self.line_edit.text()

        # Define a regular expression pattern to match integers separated by comma and space
        pattern = r'^\d+(, \d+)*$'

        if re.match(pattern, input_text):
            QtWidgets.QMessageBox.information(self, 'Valid Input', 'Input is valid!')
        else:
            QtWidgets.QMessageBox.warning(self, 'Invalid Input', 'Input must be integers separated by comma and space.')


    def create_object(self):

        if self.status > 0:

            self.show_popup('warning', 'Cannot create a new object. Please delete the previous one first.')
        
        else:

            self.buttonCreateObject.setEnabled(False)  # Disable the button
            indexes, prtf_size, objectives, start_date, end_date, MUT, CX, pop_size, generations, bb_filepath = self.get_selected_inputs()

            if self.bb_mode:
                try:
                    self.obj = self.obj_class(indexes, prtf_size, objectives, start_date, end_date, bb_path = bb_filepath)
                    self.show_popup('info', 'Object created.')
                except (TypeError, ValueError, FileNotFoundError, pickle.UnpicklingError, ImportError) as e:
                    self.show_popup('warning', f'Could not create Object. Error raised:\n{e}')

            else:
                try:
                    self.obj = self.obj_class(indexes, prtf_size, objectives, start_date, end_date, MUT, CX, pop_size, generations)
                    self.show_popup('info', 'Object created.')
                except (TypeError, ValueError, FileNotFoundError, pickle.UnpicklingError) as e:
                    self.show_popup('warning', f'Could not create Object. Error raised:\n{e}')

            self.update_status()
            self.buttonCreateObject.setEnabled(True)  # Re-enable the button


    def get_selected_inputs(self):

        prtf_size = self.spinPrtfSize.value()
        bb_filepath = self.lineInputBBFile.text()
        if len(self.lineInputObjectives.text()) != 0:
            values = re.split(r'\s*,\s*', self.lineInputObjectives.text())
            objectives = tuple(map(int, values))
        else: 
            objectives = (1, -1)
        MUTPB, CXPB = self.doubleSpinMUT.value(), self.doubleSpinCX.value()
        pop_size, generations = self.spinPopSize.value(), self.spinGenerations.value()
        start_date, end_date = str(self.startDateSelect.date().toPyDate()), str(self.endDateSelect.date().toPyDate())
        indexes = self.get_selected_indexes()

        return indexes, prtf_size, objectives, start_date, end_date, CXPB, MUTPB, pop_size, generations, bb_filepath


    def delete_object(self):

        if self.status >= 1:
            delattr(self, 'obj')
            self.show_popup('info', 'Object deleted')
            self.update_status()
        else:
            self.show_popup('warning', f'There is no {self.module} object')


    def import_object(self):

        if self.status >= 1:

            self.show_popup('warning', 'Cannot import a new object. Please delete the previous one first.')

        else:

            current_dir = os.getcwd()
            if isinstance(self, Ui_AS):
                dir = os.path.join(*[current_dir, c.prtf_folder, c.as_folder])
            elif isinstance(self, Ui_PO):
                dir = os.path.join(*[current_dir, c.prtf_folder, c.po_folder])
            
            # options = QtWidgets.QFileDialog.Options()
            # filePath, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open File", dir, "Pickle Files (*.pkl)", options=options)
            # file_no_extension = os.path.splitext(os.path.basename(filename))[0]
            filename, _ = self.open_file_dialog(dir, "Pickle Files (*.pkl)")

            if filename is None:
                self.show_popup('warning', 'Please select a valid File')
                return
            
            self.obj = self.obj_class(filename = f"{filename}.pkl")
            self.update_status()


    def open_file_dialog(self, initial_dir, filetype):

        options = QtWidgets.QFileDialog.Options()
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open File", initial_dir, filetype, options=options)

        if filePath:
            filename, extension = os.path.splitext(os.path.basename(filePath))
            return filename, extension
        else:
            return None, None


    def save_object(self):

        if self.status > 0:
            filename = self.show_input_popup("File name", "Please enter a name for the file:")
            saved_filename = self.obj.save_to_pickle(filename)
            self.show_popup('info', f'Object saved as {saved_filename}')

        else:
            self.show_popup('warning', 'Cannot save an object. Please create one first.')
        return


    def create_test_object(self):

        if self.status >= 3:
            years, months, days = self.get_test_interval_inputs()
            self.test_obj = self.obj.create_test_asset_selection(years = years, months = months, days = days)
            self.show_popup('info', 'Test Object Created.')
            self.update_test_plot_text()
        else:
            self.show_popup('warning', 'Cannot create a test object. Please select a Final Portfolio.')


    def get_test_interval_inputs(self):

        years = self.spinTestYears.value()
        months = self.spinTestMonths.value()
        days = self.spinTestDays.value()

        return years, months, days


    def populate_asset_selection_details(self):
        
        if self.status > 0:

            if self.bb_mode:
                self.labelObject.setText("Black-Box Object:")
            else:
                self.labelObject.setText("Evolutionary Object:")
            self.labelIdxs.setText("Indexes: " + str(list(self.obj.indexes.keys())))
            self.labelPrtfSizeInfo.setText("Portfolio Size: " + str(self.obj.prtf_size))
            self.labelObjectiveWeights.setText("Objective Weights: " + str(self.obj.objectives))
            self.labelStartDateInfo.setText("Start Date: " + self.obj.start_date)
            self.labelEndDateInfo.setText("End Date: " + self.obj.end_date)
            if self.bb_mode:
                self.labelBBPathInfo.setText("Black-box Filepath: " + str(self.obj.bb_path))
            else:
                self.labelCXInfo.setText("Crossover Probability: " + str(self.obj.CXPB))
                self.labelMUTInfo.setText("Mutation Probability: " + str(self.obj.MUTPB))
                self.labelPopSizeInfo.setText("Population Size: " + str(self.obj.pop_size))
                self.labelGenerationsInfo.setText("Generations: " + str(self.obj.generations))
        
        else:
            add_str = " - "
            self.labelObject.setText("There is no Object.")
            self.labelIdxs.setText("Indexes:" + add_str)
            self.labelPrtfSizeInfo.setText("Portfolio Size:" + add_str)
            self.labelObjectiveWeights.setText("Objective Weights: " + add_str)
            self.labelStartDateInfo.setText("Start Date:" + add_str)
            self.labelEndDateInfo.setText("End Date:" + add_str)
            self.labelCXInfo.setText("Crossover Probability:" + add_str)
            self.labelMUTInfo.setText("Mutation Probability:" + add_str)
            self.labelPopSizeInfo.setText("Population Size:" + add_str)
            self.labelGenerationsInfo.setText("Generations:" + add_str)
            self.labelBBPathInfo.setText("Black-box Filepath:" + add_str)


    def create_index_dropdown(self):

        self.index_dropdown()

        for i in self.index_dict:
            
            item = QtWidgets.QListWidgetItem()
            item.setText(str(i))
            self.listIndexes.addItem(item)
        
        if self.listIndexes.count() > 0:
            self.listIndexes.setCurrentRow(0)


    def index_dropdown(self):

        all_classes = [(name, cls) for name, cls in inspect.getmembers(index) if inspect.isclass(cls) and cls.__module__ == index.__name__ and name != "Index"]
        
        # Check if each class has a corresponding file in the Data folder
        available_classes = []
        for name, cls in all_classes:
            file_path = get_path(name, filename = cls.asset_list_csv_file)
            if os.path.exists(file_path):
                available_classes.append((name, cls))

        self.index_dict = {i[0]: i[1]() for i in available_classes}
        return


    #Not in current use
    def create_ss_layout(self):

        self.verticalLayoutWidget = QtWidgets.QWidget(self.tabRunAlgorithm)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(50, 60, 201, 391))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")

        self.layoutInfo = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.layoutInfo.setContentsMargins(0, 0, 0, 0)
        self.layoutInfo.setObjectName("layoutInfo")
        
        self.labelObject = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.labelObject.setObjectName("labelObject")
        self.layoutInfo.addWidget(self.labelObject)

        self.labelIdxs = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.labelIdxs.setObjectName("labelIdxs")
        self.layoutInfo.addWidget(self.labelIdxs)

        self.labelPrtfSizeInfo = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.labelPrtfSizeInfo.setObjectName("labelPrtfSizeInfo")
        self.layoutInfo.addWidget(self.labelPrtfSizeInfo)

        self.labelStartDateInfo = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.labelStartDateInfo.setObjectName("labelStartDateInfo")
        self.layoutInfo.addWidget(self.labelStartDateInfo)

        self.labelEndDateInfo = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.labelEndDateInfo.setObjectName("labelEndDateInfo")
        self.layoutInfo.addWidget(self.labelEndDateInfo)

        self.labelCXInfo = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.labelCXInfo.setObjectName("labelCXInfo")
        self.layoutInfo.addWidget(self.labelCXInfo)

        self.labelMUTInfo = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.labelMUTInfo.setObjectName("labelMUTInfo")
        self.layoutInfo.addWidget(self.labelMUTInfo)

        self.labelPopSizeInfo = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.labelPopSizeInfo.setObjectName("labelPopSizeInfo")
        self.layoutInfo.addWidget(self.labelPopSizeInfo)

        self.labelGenerationsInfo = QtWidgets.QLabel('Ora fds', self.verticalLayoutWidget)
        self.labelGenerationsInfo.setObjectName("labelGenerationsInfo")
        self.layoutInfo.addWidget(self.labelGenerationsInfo)


    def get_methods(self, file_path):
        module_name = file_path.split('.')[0]
        module = importlib.import_module(module_name)

        methods = [name for name, obj in inspect.getmembers(module) if inspect.isfunction(obj)]
        methods = [i for i in methods if i[:4] != "aux_"]

        return methods

    def remove_duplicates(self, input_list):
        seen = set()
        result = []
        indices = []

        for i, item in enumerate(input_list):
            tuple_item = tuple(item)
            if tuple_item not in seen:
                result.append(item)
                indices.append(i)
                seen.add(tuple_item)

        return result, indices


    def disable_ea_algo_tab(self):

        self.comboBoxMate.setDisabled(self.bb_mode)
        self.comboBoxMutate.setDisabled(self.bb_mode)
        self.comboBoxSelect.setDisabled(self.bb_mode)
        self.comboBoxEvaluate.setDisabled(self.bb_mode)
        self.comboBoxAlgo.setDisabled(self.bb_mode)
        self.buttonInitPop.setDisabled(self.bb_mode)
        self.buttonRegisterMethods.setDisabled(self.bb_mode)

        return

    
    def show_popup(self, message_type, message):
        """
        Shows a popup message of a specified type.

        :param message_type: The type of the message (e.g., 'info', 'warning', 'critical', 'question').
        :param message: The message to display in the popup.
        """
        msg = QtWidgets.QMessageBox()
        msg.setText(message)
        msg.setTextFormat(QtCore.Qt.PlainText)
        
        if message_type == 'info':
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setWindowTitle("Information")
        elif message_type == 'warning':
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setWindowTitle("Warning")
        elif message_type == 'critical':
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setWindowTitle("Critical")
        elif message_type == 'question':
            msg.setIcon(QtWidgets.QMessageBox.Question)
            msg.setWindowTitle("Question")
        else:
            msg.setIcon(QtWidgets.QMessageBox.NoIcon)
            msg.setWindowTitle("Message")

        msg.exec_()


    # def show_input_popup(self, title= "", label= ""):
    #     text, ok = QtWidgets.QInputDialog.getText(None, title, label)
    #     if ok:
    #         return text
    #     return None

    def show_input_popup(self, title="", label=""):
        text, ok = QtWidgets.QInputDialog.getText(None, title, label)
        if ok:
            filename = str(text).strip()
            # Check for invalid filename characters (Windows: \ / : * ? " < > |)
            if filename and not any(c in filename for c in r'\/:*?"<>|'):
                return filename
        return None

class Backend_AS(QtWidgets.QWidget, Backend, Ui_AS):

    module = 'Asset Selection'

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        Backend.__init__(self)
        Ui_AS.__init__(self)
        self.setupUi(self)


    def setup_backend(self):

        self.obj_class = AssetSelection
        self.setup_create_object()
        self.setup_run_algorithm()
        self.setup_choose_individual()
        self.setup_test_portfolios()

        self.setup_create_po_from_as()

        self.status = 0
        self.update_status()


    def update_status(self):

        #Stores previous state
        previous_state = self.status

        #If the object exists, status is 1
        if hasattr(self, 'obj'):
            self.status = 1
            #If an algorithm was ran, status is 2
            if hasattr(self.obj, 'pareto_front'):
                self.status = 2
                #If a final portfolio was chosen, status is 3
                if hasattr(self.obj, 'final_prtf'):
                    self.status = 3
        #If none of the above, status is 0
        else:
            self.status = 0

        #Unless the object had a final portfolio which was removed
        #Update text on the Choose Individual plot
        if not (previous_state == 3 and self.status == 2):
            self.update_choose_individual_plot_text()

        #If we have Test object
        self.update_test_plot_text()

        #Unless the object had a final portfolio which was removed
        #Or we selected a final portfolio
        #Update the list of portfolios on display
        if not ((previous_state == 3 and self.status == 2) or (previous_state == 2 and self.status == 3)):
            self.update_checkbox_prtfs() 

        #Populate details of Asset Selection object
        self.populate_asset_selection_details()

        self.disable_ea_algo_tab()

        return


    def setup_create_po_from_as(self):

        self.buttonCreatePO.clicked.connect(self.create_po_from_as)
    

    def create_po_from_as(self):

        if self.status != 3:
            self.show_popup('warning', 'Not possible to create a Portfolio Optimization Object.\nPlease select a Final Portfolio first.')
            return

        if self.tab_PO.status == 0:
            new_po = self.obj.create_portfolio_optimization()
            self.tab_PO.obj = new_po
            self.show_popup('info', 'Portfolio Optimization Object Created.')
            
            self.tab_PO.update_status()
        else:
            self.show_popup('warning', 'Portfolio Optimization Object already exists.')


    def update_choose_individual_plot_text(self):

        #If the object does not exist, show the text to create an object
        if self.status == 0:
            self.text_for_plot_choose_individual("To Proceed:\nCreate an Asset Selection Object")
        #If the object exists, show the text to run the algorithm
        elif self.status == 1:
            self.text_for_plot_choose_individual("To Proceed:\nRun Algorithm")
        #If the algorithm was run, show the text to choose a plot
        elif self.status == 2:
            self.text_for_plot_choose_individual('To Proceed:\nChoose a Plot')


    def update_test_plot_text(self):

        #If the object does not exist, show the text to create an object
        if not hasattr(self, 'test_obj'):
            self.text_for_plot_test("To Proceed:\nCreate a Test Object")
        #Choose a plot
        else:
            self.text_for_plot_test('To Proceed:\nLoad a Plot')


    def connect_algo_buttons(self):

        self.buttonInitPop.clicked.connect(self.backend_init_pop)
        self.buttonRegisterMethods.clicked.connect(self.backend_register_methods)
        self.buttonRunAlgo.clicked.connect(self.backend_run_algo)


class Backend_PO(Backend, Ui_PO):

    module = 'Portfolio Optimization'


    def setup_backend(self):

        self.obj_class = PortfolioOptimization
        self.setup_create_object()
        self.setup_run_algorithm()
        self.setup_choose_individual()

        self.status = 0
        self.update_status()


    def update_status(self):

        previous_state = self.status
        if hasattr(self, 'obj'):
            self.status = 1
            if hasattr(self.obj, 'pareto_front'):
                self.status = 2
                if hasattr(self.obj, 'final_prtf'):
                    self.status = 3
        else:
            self.status = 0

        if not (previous_state == 3 and self.status == 2):
            self.update_choose_individual_plot_text()

        if not ((previous_state == 3 and self.status == 2) or (previous_state == 2 and self.status == 3)):
            self.update_checkbox_prtfs() 

        self.populate_asset_selection_details()
        self.disable_ea_algo_tab()

        return
    
    # def update_choose_individual_plot_text(self):

    #     if self.status == 0:
    #         self.text_for_plot_choose_individual("Create a Portfolio Optimization\n Object")
    #         if self.status == 1:
    #             self.text_for_plot_choose_individual("Run Algorithm")
    #             if self.status == 2:
    #                 self.text_for_plot_choose_individual('Choose a Plot')


    def update_choose_individual_plot_text(self):

        #If the object does not exist, show the text to create an object
        if self.status == 0:
            self.text_for_plot_choose_individual("To Proceed:\nCreate a Portfolio Optimization Object")
        #If the object exists, show the text to run the algorithm
        elif self.status == 1:
            self.text_for_plot_choose_individual("To Proceed:\nRun Algorithm")
        #If the algorithm was run, show the text to choose a plot
        elif self.status == 2:
            self.text_for_plot_choose_individual('To Proceed:\nChoose a Plot')


    def connect_algo_buttons(self):

        self.buttonInitPop.clicked.connect(self.backend_init_pop)
        self.buttonRegisterMethods.clicked.connect(self.backend_register_methods)
        self.buttonRunAlgo.clicked.connect(self.backend_run_algo)
        self.buttonImportAssetsPRTF.clicked.connect(lambda: self.backend_import_assets('pkl'))
        self.buttonImportAssetsJSON.clicked.connect(lambda: self.backend_import_assets('json'))

    def backend_import_assets(self, ftype):

        if self.status == 1:

            current_dir = os.getcwd()
            if ftype == 'pkl':
                dir = os.path.join(*[current_dir, c.prtf_folder, c.as_folder])
                extension = "Pickle Files (*.pkl)"
            if ftype == 'json':
                dir = os.path.join(*[current_dir, c.prtf_folder, c.assets_folder])
                extension = "JSON Files (*.json)"
                
            filename, _ = self.open_file_dialog(dir, extension)
            if filename is None:
                print('Please choose a valid file.')
                return
            
            if 'pkl' in extension:
                self.obj.set_assets(pkl_filename = filename)
            if 'json' in extension:
                self.obj.set_assets(json_filename = filename)



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Backend()
    ui.setup_main_tab(MainWindow)
    # ui.setupUi(MainWindow)
    # ui.setup_backend_as()
    MainWindow.show()
    sys.exit(app.exec_())