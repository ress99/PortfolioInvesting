# pylint: skip-file

from main_tab import Ui_MainWindow
from backend import Backend_AS, Backend_PO
from PyQt5 import QtCore, QtGui, QtWidgets
import logging
import config_logger # pylint: disable=unused-import


logger = logging.getLogger(__name__)
logger.info("Starting GUI App")

class GUI_(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(GUI_, self).__init__(parent)

        # Setup the main window layout
        self.setupUi(self)

        # Create an instance of Backend_AS
        self.tab_AS = Backend_AS()
        self.tab_AS.setupUi(self.tabAssetSel)
        self.tab_AS.setup_backend()

        self.tab_PO = Backend_PO()
        self.tab_PO.setupUi(self.tabPortOpt)
        self.tab_PO.setup_backend()

        self.tab_AS.tab_PO = self.tab_PO
        self.tab_PO.tab_AS = self.tab_AS


class GUI(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.gui_inits()

    def gui_inits(self):
        self.tab_AS_instance = Backend_AS()
        
        layout = QtWidgets.QVBoxLayout()  # create a new vertical layout
        self.tabAssetSel.setLayout(layout)  # set this layout to the tab
        layout.addWidget(self.tab_AS_instance)

    def gui_init(self):

        self.tab_AS = Backend_AS()

        # Add Backend_AS to the vertical layout of the tabAssetSel tab
        self.verticalLayout_3.addWidget(self.tab_AS)

        # self.tab_AS = Backend_AS()
        # self.tab_AS.setupUi(self.tabAssetSel)
        # self.tab_AS.setup_backend()


        self.tab_PO = Backend_PO()
        self.tab_PO.setupUi(self.tabPortOpt)
        self.tab_PO.setup_backend()


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = GUI_()
    window.show()
    sys.exit(app.exec_())



# if __name__ == "__main__":
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     MainWindow = QtWidgets.QMainWindow()
#     ui = GUI()
#     # ui.setupUi(MainWindow)
#     # ui.gui_init()
#     MainWindow.show()
#     sys.exit(app.exec_())

# if __name__ == "__main__":
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     # MainWindow = QtWidgets.QMainWindow()
#     Form = QtWidgets.QWidget()
#     ui = GUI(Form)

#     Form.show()
#     sys.exit(app.exec_())


    # def setup_main_tab(self, MainWindow):

    #     # MainWindow = QtWidgets.QMainWindow()
    #     self.mw = Ui_MainWindow()
    #     self.mw.setupUi(MainWindow)

    #     self.AS = Ui_AS()
    #     self.AS.setupUi(self.mw.tabAssetSel)
    #     self.AS.setup_backend_as()

    #     self.PO = Ui_PO()
    #     self.PO.setupUi(self.mw.tabPortOpt)
    #     # self.PO.setup_backend_as()