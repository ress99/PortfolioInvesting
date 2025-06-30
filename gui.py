# pylint: skip-file

from main_tab import Ui_MainWindow
from backend import Backend_AS, Backend_PO
from PyQt5 import QtWidgets
import logging
import config_logger # pylint: disable=unused-import

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = GUI_()
    window.show()
    sys.exit(app.exec_())
